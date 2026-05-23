from __future__ import annotations


def load_ipython_extension(ipython: Any) -> None:
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def omniff(line: str, cell: str) -> None:
        from omniff.runtime.config import OmniFFConfig, RouterConfig
        from omniff.runtime.engine import OmniFFRuntime
        from IPython.display import display, Image, Audio, Markdown

        config = OmniFFConfig(
            name="omniff", version="1.0",
            router=RouterConfig(router_type="keyword", path=""),
        )
        runtime = OmniFFRuntime(config)

        parts = line.strip().split()
        kwargs = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                kwargs[k] = v

        result = runtime.run(
            input=cell.strip(),
            thinking=kwargs.get("thinking", "normal"),
            output_modality=kwargs.get("output", None),
        )

        if result.output_path:
            path = result.output_path
            if path.endswith((".png", ".jpg", ".jpeg", ".webp")):
                display(Image(filename=path))
            elif path.endswith((".wav", ".mp3")):
                display(Audio(filename=path))
            else:
                print(f"Output saved: {path}")
        elif result.output_text:
            display(Markdown(result.output_text))


from typing import Any
