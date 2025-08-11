import argparse
import json

from meomaya.core.modelify import Modelify


def main():
    """CLI for MeoMaya Modelify engine (multimodal).

    Examples:
      python -m meomaya.cli.meomaya_cmd "Hello world" --mode text
      python -m meomaya.cli.meomaya_cmd path/to/image.jpg  # auto-detects image
    """

    parser = argparse.ArgumentParser(description="MeoMaya Modelify (multimodal)")
    parser.add_argument("input", type=str, help="Input text or file path")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Override auto-detected mode: text, audio, image, video, 3d, fusion",
    )
    parser.add_argument(
        "--model", type=str, default="default", help="Model name (placeholder)"
    )

    args = parser.parse_args()

    engine = Modelify(mode=args.mode, model=args.model)
    result = engine.run(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
