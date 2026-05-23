from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", required=True, help="Input file or text")
    parser.add_argument("-p", "--prompt", help="Prompt / instruction")
    parser.add_argument("-f", "--of", dest="output_format", help="Output format: text, image, video, audio")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--thinking", default="normal", help="Thinking level: off, fast, normal, deep")
    parser.add_argument("--strength", type=float, help="Style/edit strength 0.0-1.0")
    parser.add_argument("--lang", help="Language hint")
    parser.add_argument("--model", default="auto", help="Model override")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--config", default="omniff.yaml", help="Config file path")
    parser.add_argument("--negative-prompt", help="Negative prompt for generation")


def _cmd_run(args: argparse.Namespace) -> None:
    from omniff.runtime.config import OmniFFConfig, RouterConfig
    from omniff.runtime.engine import OmniFFRuntime

    config_path = Path(args.config)
    if config_path.exists():
        runtime = OmniFFRuntime.from_yaml(config_path)
    else:
        config = OmniFFConfig(
            name="omniff",
            version="0.1",
            router=RouterConfig(router_type="keyword", path=""),
        )
        runtime = OmniFFRuntime(config)

    controls: dict = {}
    if args.model != "auto":
        controls["model_id"] = args.model
    if args.strength is not None:
        controls["strength"] = args.strength
    if args.lang:
        controls["language"] = args.lang
    if args.seed is not None:
        controls["seed"] = args.seed
    if args.negative_prompt:
        controls["negative_prompt"] = args.negative_prompt

    result = runtime.run(
        input=args.input,
        prompt=args.prompt,
        output_modality=args.output_format,
        thinking=args.thinking,
        controls=controls,
        output=args.output,
    )

    if result.output_text:
        print(result.output_text)
    if result.output_path:
        print(f"Output saved: {result.output_path}")
    if args.output and result.output_text:
        result.save(args.output)


def _cmd_doctor(args: argparse.Namespace) -> None:
    import shutil
    import platform

    print("OmniFF Doctor")
    print("=" * 40)

    print(f"\nPython: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem
                free = mem - torch.cuda.memory_reserved(i)
                print(f"  GPU {i}: {name} ({mem / 1024**3:.1f} GB total)")
        else:
            print("  GPU: not available (CPU only)")
    except ImportError:
        print("\nPyTorch: NOT INSTALLED")

    deps = {
        "transformers": "transformers",
        "diffusers": "diffusers",
        "pydantic": "pydantic",
        "soundfile": "soundfile",
        "cv2": "opencv",
        "scipy": "scipy",
        "fastapi": "fastapi",
        "gradio": "gradio",
    }
    print("\nDependencies:")
    for mod, label in deps.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "ok")
            print(f"  {label}: {ver}")
        except ImportError:
            print(f"  {label}: NOT INSTALLED")

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        models = [d.name for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("models--")]
        if models:
            print(f"\nCached models ({len(models)}):")
            for m in sorted(models):
                name = m.replace("models--", "").replace("--", "/")
                print(f"  {name}")
        else:
            print("\nCached models: none")
    else:
        print("\nHuggingFace cache: not found")

    print("\nStatus: OK")


def _cmd_models(args: argparse.Namespace) -> None:
    action = args.action

    if action == "list":
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if not cache_dir.exists():
            print("No cached models found.")
            return
        models = sorted(
            d.name for d in cache_dir.iterdir()
            if d.is_dir() and d.name.startswith("models--")
        )
        if not models:
            print("No cached models found.")
            return
        print(f"Cached models ({len(models)}):")
        for m in models:
            name = m.replace("models--", "").replace("--", "/")
            model_dir = cache_dir / m
            size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            print(f"  {name}  ({size / 1024**3:.1f} GB)")

    elif action == "pull":
        if not args.model_id:
            print("Error: --model-id required for pull")
            sys.exit(1)
        try:
            from huggingface_hub import snapshot_download
            print(f"Downloading {args.model_id}...")
            path = snapshot_download(args.model_id)
            print(f"Downloaded to: {path}")
        except ImportError:
            print("Error: pip install huggingface_hub")
            sys.exit(1)

    elif action == "remove":
        if not args.model_id:
            print("Error: --model-id required for remove")
            sys.exit(1)
        import shutil
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        dir_name = "models--" + args.model_id.replace("/", "--")
        model_dir = cache_dir / dir_name
        if model_dir.exists():
            size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            shutil.rmtree(model_dir)
            print(f"Removed {args.model_id} ({size / 1024**3:.1f} GB freed)")
        else:
            print(f"Model not found in cache: {args.model_id}")
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="omniff",
        description="OmniFF — FFmpeg-like multimodal AI runtime",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run inference pipeline")
    _add_run_args(run_parser)

    sub.add_parser("doctor", help="Check system environment and dependencies")

    models_parser = sub.add_parser("models", help="Manage cached models")
    models_parser.add_argument("action", choices=["list", "pull", "remove"])
    models_parser.add_argument("--model-id", help="Model ID (e.g., Qwen/Qwen3-4B)")

    args, remaining = parser.parse_known_args()

    if args.command == "doctor":
        _cmd_doctor(args)
    elif args.command == "models":
        _cmd_models(args)
    elif args.command == "run":
        _cmd_run(args)
    else:
        run_parser_direct = argparse.ArgumentParser(prog="omniff", parents=[], add_help=False)
        _add_run_args(run_parser_direct)
        try:
            run_args = run_parser_direct.parse_args(sys.argv[1:])
            _cmd_run(run_args)
        except SystemExit:
            parser.print_help()


if __name__ == "__main__":
    main()
