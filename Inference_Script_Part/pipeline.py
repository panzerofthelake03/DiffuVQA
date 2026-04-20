import json
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from diffuvqa.rounding import denoised_fn_round
from shared.basic_utils import create_model_and_diffusion, load_tokenizer

from .config_utils import build_runtime_args
from .preprocess import build_condition_tensors, load_image_tensor


class PipelineLoadError(RuntimeError):
    pass


@dataclass
class InferenceResult:
    checkpoint_path: str
    image_path: str
    question: str
    answer: str
    answer_tokens: list[int]
    latency_seconds: float
    decode_mode: str
    sampling_steps: int
    confidence: float
    confidence_non_pad: float
    avg_token_logprob: float
    pad_ratio: float
    non_pad_token_count: int
    quality_flag: str
    timestamp_utc: str
    error: Optional[str] = None


class DiffuVQAInferencePipeline:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        strict_load: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = self._resolve_device(device)
        self.args = build_runtime_args(self.checkpoint_path, overrides=overrides)
        self.strict_load = strict_load

        self.model = None
        self.diffusion = None
        self.tokenizer = None
        self.model_emb = None
        self.loaded = False

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _extract_state_dict(raw_ckpt: Any) -> Dict[str, torch.Tensor]:
        if isinstance(raw_ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model", "ema"):
                value = raw_ckpt.get(key)
                if isinstance(value, dict):
                    return value
            if all(isinstance(v, torch.Tensor) for v in raw_ckpt.values()):
                return raw_ckpt
        raise PipelineLoadError("Unsupported checkpoint format: no state_dict found")

    @staticmethod
    def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    def _build_rounding_embedding(self) -> torch.nn.Embedding:
        word_embedding = self.model.word_embedding
        emb_weight = word_embedding.weight.detach()

        if hasattr(self.model, "get_embeds"):
            with torch.no_grad():
                all_ids = torch.arange(emb_weight.size(0), device=self.device, dtype=torch.long)
                projected = self.model.get_embeds(all_ids).detach()
            model_emb = torch.nn.Embedding.from_pretrained(projected, freeze=True)
        else:
            model_emb = torch.nn.Embedding.from_pretrained(emb_weight, freeze=True)

        return model_emb.to(self.device).eval()

    def load(self):
        try:
            self.model, self.diffusion = create_model_and_diffusion(args=self.args)
            raw_ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = self._strip_module_prefix(self._extract_state_dict(raw_ckpt))

            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                if self.strict_load:
                    raise
                # Fallback mode supports older/newer checkpoints with minor key drift.
                self.model.load_state_dict(state_dict, strict=False)

            self.model = self.model.to(self.device).eval().requires_grad_(False)
            self.tokenizer = load_tokenizer(self.args)
            self.model_emb = self._build_rounding_embedding()
            self.loaded = True
        except Exception as exc:
            raise PipelineLoadError(f"Failed to load inference pipeline: {exc}") from exc

        return self

    def _sampling_settings(self, requested_steps: Optional[int]) -> tuple[Any, int, str]:
        total_steps = int(self.args.diffusion_steps)
        if requested_steps is None:
            requested_steps = total_steps

        requested_steps = int(requested_steps)
        if requested_steps >= total_steps:
            return self.diffusion.p_sample_loop, 1, "p_sample"

        step_gap = max(total_steps // requested_steps, 1)
        return self.diffusion.ddim_sample_loop, step_gap, "ddim"

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = " ".join((text or "").replace("\n", " ").split())
        return cleaned.strip()

    @staticmethod
    def _mean_confidence_from_logits(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[float, float]:
        probs = torch.softmax(logits, dim=-1)
        chosen = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

        if valid_mask is None:
            confidence = float(chosen.mean().item())
            avg_logprob = float(torch.log(chosen.clamp(min=1e-12)).mean().item())
            return confidence, avg_logprob

        valid_mask = valid_mask.to(chosen.dtype)
        valid_count = float(valid_mask.sum().item())
        if valid_count <= 0:
            return 0.0, 0.0

        confidence = float((chosen * valid_mask).sum().item() / valid_count)
        avg_logprob = float((torch.log(chosen.clamp(min=1e-12)) * valid_mask).sum().item() / valid_count)
        return confidence, avg_logprob

    def predict(
        self,
        image_path: str,
        question: str,
        answer_seed_text: str = "",
        sampling_steps: Optional[int] = None,
        top_p: float = 0.0,
        clamp_step: int = 0,
        clip_denoised: bool = False,
    ) -> InferenceResult:
        if not self.loaded:
            self.load()

        started = time.time()
        try:
            image_tensor = load_image_tensor(
                image_path=image_path,
                image_resolution=int(self.args.image_resolution),
                device=self.device,
            )

            cond, cond_for_ddpm = build_condition_tensors(
                tokenizer_obj=self.tokenizer,
                question=question,
                seq_len=int(self.args.seq_len),
                device=self.device,
                answer_seed_text=answer_seed_text,
                image_name=Path(image_path).name,
            )

            input_ids_a = cond["input_a_id"]
            input_emb = self.model.get_embeds(input_ids_a)
            fuse_feats, _ = self.model.get_ddpm_input(image_tensor, cond_for_ddpm)
            x_start = torch.cat([fuse_feats, input_emb], dim=1)

            # Build the generation mask aligned with x_start = [fuse(N) | answer(A)].
            # The fuse region is fixed (mask=0); the answer region must be noised (mask=1)
            # so the diffusion loop actually generates tokens there.
            # NOTE: text_mask from cond["input_mask"] covers Q+A (2*seq_len), but
            # x_start only contains [fuse | answer_emb], so we build the mask directly.
            bsz = fuse_feats.size(0)
            fuse_len = fuse_feats.size(1)
            answer_len_mask = input_emb.size(1)
            full_mask = torch.cat([
                torch.zeros((bsz, fuse_len), dtype=torch.long, device=self.device),
                torch.ones((bsz, answer_len_mask), dtype=torch.long, device=self.device),
            ], dim=1)  # shape: (bsz, fuse_len + answer_len)

            input_ids_mask = torch.broadcast_to(full_mask.unsqueeze(-1), x_start.shape)

            noise = torch.randn_like(x_start)
            x_noised = torch.where(input_ids_mask == 0, x_start, noise)

            sample_fn, step_gap, decode_mode = self._sampling_settings(sampling_steps)
            sample_shape = tuple(x_start.shape)

            samples = sample_fn(
                self.model,
                sample_shape,
                noise=x_noised,
                clip_denoised=clip_denoised,
                denoised_fn=partial(denoised_fn_round, self.args, self.model_emb),
                model_kwargs={},
                top_p=top_p,
                clamp_step=int(clamp_step),
                clamp_first=True,
                mask=input_ids_mask,
                x_start=x_start,
                gap=step_gap,
            )

            sample = samples[-1]
            answer_len = input_emb.size(1)
            answer_hidden = sample[:, -answer_len:, :]

            logits = self.model.get_logits(answer_hidden)
            chosen = torch.topk(logits, k=1, dim=-1).indices.squeeze(-1)

            answer_text = self._clean_text(self.tokenizer.decode_token(chosen[0].unsqueeze(-1).cpu()))
            confidence, avg_logprob = self._mean_confidence_from_logits(logits, chosen)
            pad_token_id = int(self.tokenizer.pad_token_id)
            non_pad_mask = (chosen != pad_token_id)
            non_pad_count = int(non_pad_mask.sum().item())
            total_count = int(chosen.numel())
            pad_ratio = float((total_count - non_pad_count) / total_count) if total_count > 0 else 1.0
            confidence_non_pad, _ = self._mean_confidence_from_logits(
                logits,
                chosen,
                valid_mask=non_pad_mask,
            )

            if non_pad_count == 0:
                quality_flag = "low_quality_pad_only"
            elif not answer_text:
                quality_flag = "low_quality_empty_after_cleanup"
            else:
                quality_flag = "ok"

            return InferenceResult(
                checkpoint_path=str(self.checkpoint_path),
                image_path=str(Path(image_path).resolve()),
                question=question,
                answer=answer_text,
                answer_tokens=chosen[0].tolist(),
                latency_seconds=round(time.time() - started, 4),
                decode_mode=decode_mode,
                sampling_steps=int(sampling_steps or self.args.diffusion_steps),
                confidence=confidence,
                confidence_non_pad=confidence_non_pad,
                avg_token_logprob=avg_logprob,
                pad_ratio=pad_ratio,
                non_pad_token_count=non_pad_count,
                quality_flag=quality_flag,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                error=None,
            )
        except Exception as exc:
            return InferenceResult(
                checkpoint_path=str(self.checkpoint_path),
                image_path=str(Path(image_path).resolve()),
                question=question,
                answer="",
                answer_tokens=[],
                latency_seconds=round(time.time() - started, 4),
                decode_mode="error",
                sampling_steps=int(sampling_steps or self.args.diffusion_steps),
                confidence=0.0,
                confidence_non_pad=0.0,
                avg_token_logprob=0.0,
                pad_ratio=1.0,
                non_pad_token_count=0,
                quality_flag="error",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def append_result_jsonl(result: InferenceResult, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
