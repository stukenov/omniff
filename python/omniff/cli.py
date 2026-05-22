from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="omniff",
        description="OmniFF — FFmpeg-like multimodal AI runtime",
    )
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

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
