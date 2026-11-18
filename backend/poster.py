from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from typing import Any

PILLOW_AVAILABLE = False
try:
    from PIL import Image, ImageDraw, ImageFont

    PILLOW_AVAILABLE = True
except ImportError:
    pass


@dataclass
class PosterLayer:
    id: str
    type: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    text: str = ""
    fontSize: int = 48
    color: str = "#000000"
    fill: str = "#000000"
    url: str = ""
    data: str = ""
    visible: bool = True


@dataclass
class PosterTemplate:
    width: int = 1080
    height: int = 1350
    layers: list[PosterLayer] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PosterTemplate:
        layers = []
        for layer in raw.get("layers", []):
            layers.append(
                PosterLayer(
                    id=str(layer.get("id", "")),
                    type=str(layer.get("type", "rect")),
                    x=int(layer.get("x", 0)),
                    y=int(layer.get("y", 0)),
                    width=int(layer.get("width", 0)),
                    height=int(layer.get("height", 0)),
                    text=str(layer.get("text", "")),
                    fontSize=int(layer.get("fontSize", 48)),
                    color=str(layer.get("color", "#000000")),
                    fill=str(layer.get("fill", layer.get("color", "#000000"))),
                    url=str(layer.get("url", "")),
                    data=str(layer.get("data", "")),
                    visible=bool(layer.get("visible", True)),
                )
            )
        return cls(width=int(raw.get("width", 1080)), height=int(raw.get("height", 1350)), layers=layers)


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _load_font(size: int) -> Any:
    if not PILLOW_AVAILABLE:
        return None
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_poster(tpl: PosterTemplate, scale: int = 2) -> bytes:
    """Render a poster template to PNG bytes. Pure CPU, no GPU."""
    if not PILLOW_AVAILABLE:
        raise RuntimeError("Pillow not available; cannot render poster")

    w = tpl.width * scale
    h = tpl.height * scale
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for layer in tpl.layers:
        if not layer.visible:
            continue
        lx = layer.x * scale
        ly = layer.y * scale

        if layer.type == "rect":
            draw.rectangle(
                [lx, ly, lx + layer.width * scale, ly + layer.height * scale],
                fill=_hex_to_rgb(layer.fill),
            )
        elif layer.type in ("text", "price"):
            font = _load_font(layer.fontSize * scale)
            color = _hex_to_rgb(layer.color)
            draw.text((lx, ly), layer.text, fill=color, font=font)
        elif layer.type == "qr":
            placeholder = Image.new("RGB", (layer.width * scale, layer.height * scale), (0, 0, 0))
            img.paste(placeholder, (lx, ly))
        elif layer.type == "image":
            if not layer.url:
                continue
            try:
                buf = io.BytesIO()
                with urllib.request.urlopen(layer.url, timeout=10) as resp:
                    buf.write(resp.read())
                src = Image.open(buf).convert("RGB")
                if layer.width > 0 and layer.height > 0:
                    src = src.resize((layer.width * scale, layer.height * scale), Image.LANCZOS)
                img.paste(src, (lx, ly))
            except OSError:
                pass

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def render_poster_from_json(raw: dict[str, Any]) -> bytes:
    tpl = PosterTemplate.from_dict(raw)
    return render_poster(tpl)


def poster_meta(raw: dict[str, Any]) -> dict[str, Any]:
    tpl = PosterTemplate.from_dict(raw)
    return {
        "width": tpl.width,
        "height": tpl.height,
        "layer_count": len(tpl.layers),
        "visible_layers": sum(1 for layer in tpl.layers if layer.visible),
        "types": sorted({layer.type for layer in tpl.layers}),
    }
