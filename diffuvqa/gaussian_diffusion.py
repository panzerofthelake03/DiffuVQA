"""
Gaussian diffusion utilities: forward/reverse process, loss computation, DDIM sampling.
Ported from Ho et al. (2020) and extended for conditional text generation.
"""

import enum
import math

import numpy as np
import torch as th
import sys

sys.path.append('.')

import torch.nn.functional as F
import torch
import os
from .utils.nn import mean_flat
from .utils.losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Discretizes alpha_t_bar from t=0 (left-shifted interval)."""
    betas = []
    betas.append(min(1 - alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Discretizes alpha_t_bar — cumulative product of (1-beta) over t in [0,1]."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """Forward/reverse diffusion process and loss computation for conditional text generation."""

    def __init__(
            self,
            *,
            betas,
            predict_xstart,
            rescale_learned_sigmas,
            learn_sigmas,
            sigma_small,
            use_kl,
            rescale_timesteps=False,
    ):
        self.rescale_timesteps = rescale_timesteps
        self.predict_xstart = predict_xstart
        self.rescale_learned_sigmas = rescale_learned_sigmas
        self.learn_sigmas = learn_sigmas
        self.sigma_small = sigma_small
        self.use_kl = use_kl

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

        self.mapping_func = None
        self.add_mask_noise = False

    def training_losses(self, model, *args, **kwargs):
        self.model = model
        return self.training_losses_seq2seq(model, *args, **kwargs)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_start, t):
        """Distribution q(x_t | x_0): returns (mean, variance, log_variance)."""
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, f, t, noise=None, mask=None, add_information=None):
        """Sample from q(x_t | x_0). If add_information and f is set, mixes conditioning into noise (CIGN)."""
        if noise is not None and add_information and f is not None:

            noise = (
                    _extract_into_tensor(self.sqrt_alphas_cumprod, t, f.shape) * f
                    + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, f.shape)
                    * noise
            )

        elif noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return th.where(mask == 0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Posterior q(x_{t-1} | x_t, x_0): returns (mean, variance, log_variance_clipped)."""
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """Returns p(x_{t-1} | x_t) as dict with keys: mean, variance, log_variance, pred_xstart."""
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.size(0), x.size(-1)
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.predict_xstart:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None, mask=None, x_start=None,
    ):
        """One reverse diffusion step: samples x_{t-1} ~ p(x_{t-1} | x_t)."""
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        if mask is not None:
            sample = th.where(mask == 0, x_start, sample)

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"],
            "out": out
        }

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            top_p=None,
            clamp_step=None,
            clamp_first=None,
            mask=None,
            x_start=None,
            gap=1,
    ):
        """Full reverse diffusion loop from T→0; returns list of samples."""
        final = []
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                top_p=top_p,
                clamp_step=clamp_step,
                clamp_first=clamp_first,
                mask=mask,
                x_start=x_start
        ):
            final.append(sample['sample'])
        return final

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            top_p=None,
            clamp_step=None,
            clamp_first=None,
            mask=None,
            x_start=None,
    ):
        """Generator yielding p_sample output dicts for each timestep T→0."""
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur = None
                else:
                    denoised_fn_cur = denoised_fn
            else:
                if i >= clamp_step:
                    denoised_fn_cur = denoised_fn
                else:
                    denoised_fn_cur = None
            with th.no_grad():
                out = self.p_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn_cur,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]

    def _get_x_start(self, x_start_mean, std):
        """x_0 = embedding_mean + std * noise"""
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return x_start_mean + std * noise

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None):
        """-log p(w | z_0): cross-entropy loss, masked to real answer tokens if mask provided."""
        logits = get_logits(x_t)
        loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        if mask is not None:
            decoder_nll *= mask
            return decoder_nll.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        return decoder_nll.mean(dim=-1)

    def _x0_helper(self, model_output, x, t):

        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else:  # predict eps
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {'pred_xprev': pred_prev, 'pred_xstart': pred_xstart}

    def training_losses_seq2seq(self, model, image, t, model_kwargs=None, noise=None):
        """
        Training loss for one timestep.
        loss = MSE(ans_emb, denoised_ans) + NLL(denoised_ans → vocab) + pre_answer_loss (optional)
        """
        assert 'input_ids' in model_kwargs
        input_ids_x = model_kwargs.pop('input_ids').to(t.device)
        input_ids_a = model_kwargs['input_a_id'].to(t.device)
        mask = model_kwargs.pop('input_mask').to(t.device)

        # Unwrap DataParallel / custom wrappers to reach the raw model API.
        real_model = model
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'module'):
                real_model = model.model.module
            elif hasattr(model, 'module'):
                real_model = model.module
            elif hasattr(model, 'model') and hasattr(model.model, 'get_ddpm_input'):
                real_model = model.model
        except Exception:
            real_model = model

        ddpm_input_pre, ans_emb_pre = real_model.get_ddpm_input(image, model_kwargs)

        _model_args = getattr(real_model, 'args', None)
        use_noising_f         = bool(getattr(_model_args, 'use_noising_f', False))
        pre_answer_loss_weight = float(getattr(_model_args, 'pre_answer_loss_weight', 0.0))

        ans_emb = real_model.get_embeds(input_ids_a)
        x_start_mean = ans_emb
        cond_x_start_mean = torch.cat([ddpm_input_pre, x_start_mean], dim=1)

        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        x_start = self._get_x_start(x_start_mean, std)

        cond_x_start = torch.cat([ddpm_input_pre, x_start], dim=1)

        if noise is None:
            noise = th.randn_like(cond_x_start)

        # use_noising_f=False (default): pure noise start, training/inference aligned.
        # use_noising_f=True: CIGN — anchors noise toward answer manifold (ablation).
        f = cond_x_start_mean if use_noising_f else None

        # Prepend fuse_token zeros to mask so shape matches [B, fuse_len + seq_len].
        mask_to_use = mask
        if mask is not None and mask.shape[1] != cond_x_start.shape[1]:
            fuse_token_len = cond_x_start.shape[1] - mask.shape[1]
            if fuse_token_len > 0:
                fuse_pad = torch.zeros(
                    (mask.shape[0], fuse_token_len), dtype=mask.dtype, device=mask.device
                )
                mask_to_use = torch.cat([fuse_pad, mask], dim=1)
            else:
                mask_to_use = mask[:, :cond_x_start.shape[1]]

        x_t = self.q_sample(cond_x_start, f, t, noise=noise, mask=mask_to_use.to(x_start.device),
                            add_information=use_noising_f)

        get_logits = real_model.get_logits

        terms = {}

        fuse_len = ddpm_input_pre.shape[1]
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        assert model_output.shape == cond_x_start.shape
        ans_output = model_output[:, fuse_len:, :]

        # Mask PAD positions from loss so model learns to stop at SEP/PAD naturally.
        pad_id = getattr(_model_args, 'pad_token_id', 0)
        ans_len_mask = (input_ids_a != pad_id).float()  # [B, seq_len]
        ans_len_mask_3d = ans_len_mask.unsqueeze(-1)    # [B, seq_len, 1]
        ans_den = ans_len_mask.sum(dim=-1).clamp(min=1.0)  # [B]

        mse_per_token = ((ans_emb - ans_output) ** 2).mean(dim=-1)  # [B, seq_len]
        terms["mse"] = (mse_per_token * ans_len_mask).sum(dim=-1) / ans_den

        cond_model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart']
        model_out_x_start = cond_model_out_x_start[:, fuse_len:, :]

        t0_mask = (t == 0)
        t0_per_token = ((x_start_mean - model_out_x_start) ** 2).mean(dim=-1)
        t0_loss = (t0_per_token * ans_len_mask).sum(dim=-1) / ans_den
        terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])

        out_mean, _, _ = self.q_mean_variance(x_start_mean, th.LongTensor([self.num_timesteps - 1]).to(x_start_mean.device))
        tT_loss = mean_flat(out_mean ** 2).clamp(max=100.0)

        decoder_nll = self._token_discrete_loss(x_start_mean, get_logits, input_ids_a, mask=ans_len_mask)
        terms["nll"] = self._token_discrete_loss(model_out_x_start, get_logits, input_ids_a, mask=ans_len_mask)

        if pre_answer_loss_weight > 0.0:
            pre_len = min(ans_emb_pre.size(1), x_start_mean.size(1))
            pre_per_token = ((ans_emb_pre[:, :pre_len, :] - x_start_mean[:, :pre_len, :]) ** 2).mean(dim=-1)
            pre_den = ans_len_mask[:, :pre_len].sum(dim=-1).clamp(min=1.0)
            pre_answer_loss = pre_answer_loss_weight * (pre_per_token * ans_len_mask[:, :pre_len]).sum(dim=-1) / pre_den
        else:
            pre_answer_loss = th.zeros_like(terms["mse"])

        terms["loss"] = terms["mse"] + terms["nll"] + pre_answer_loss
        terms["loss"] = th.nan_to_num(terms["loss"], nan=0.0, posinf=100.0, neginf=0.0)

        return terms

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
            langevin_fn=None,
            mask=None,
            x_start=None
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            sample = langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x)

        if mask is not None:
            sample = th.where(mask == 0, x_start, sample)

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = (
                      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                      - out["pred_xstart"]
              ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            top_p=None,
            clamp_step=None,
            clamp_first=None,
            mask=None,
            x_start=None,
            gap=1,
    ):
        """DDIM sampling loop; gap controls stride between timesteps. Same usage as p_sample_loop()."""
        final = []
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                mask=mask,
                x_start=x_start,
                gap=gap
        ):
            final.append(sample['sample'])
        return final

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            langevin_fn=None,
            mask=None,
            x_start=None,
            gap=1
    ):
        """Generator yielding ddim_sample output dicts for each timestep T→0."""
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1][::gap]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):
    """
    Subsample diffusion timesteps for DDIM or strided sampling.
    section_counts: comma-separated counts per equal-sized section, or "ddimN" for DDIM stride.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]

        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)

        return self.model(x, new_ts)

