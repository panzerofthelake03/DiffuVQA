import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="DiffuVQA single image-question inference")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument("--answer-seed", default="", help="Optional seed answer text")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda or cpu")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps override")
    parser.add_argument("--top-p", type=float, default=0.0, help="Nucleus sampling p")
    parser.add_argument("--clamp-step", type=int, default=0, help="Clamp timestep")
    parser.add_argument("--clip-denoised", action="store_true", help="Enable diffusion clip_denoised")
    parser.add_argument("--output-jsonl", default="", help="Optional JSONL log output path")
    parser.add_argument(
        "--fail-on-low-quality",
        action="store_true",
        help="Exit with non-zero status when quality_flag is low_quality_*",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Require exact checkpoint/model key match",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Import lazily so `--help` works even when heavy runtime deps are absent.
    if __package__:
        from .pipeline import DiffuVQAInferencePipeline
    else:
        repo_root = Path(__file__).resolve().parent.parent
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        from Inference_Script_Part.pipeline import DiffuVQAInferencePipeline

    pipeline = DiffuVQAInferencePipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
        strict_load=args.strict_load,
    ).load()

    result = pipeline.predict(
        image_path=args.image,
        question=args.question,
        answer_seed_text=args.answer_seed,
        sampling_steps=args.steps,
        top_p=args.top_p,
        clamp_step=args.clamp_step,
        clip_denoised=args.clip_denoised,
    )

    if args.output_jsonl:
        pipeline.append_result_jsonl(result, args.output_jsonl)

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))

    if args.fail_on_low_quality and str(result.quality_flag).startswith("low_quality_"):
        print(
            f"Low quality generation detected ({result.quality_flag}); exiting with status 2.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
