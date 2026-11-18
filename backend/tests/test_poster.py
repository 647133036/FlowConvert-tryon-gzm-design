from poster import PosterTemplate, _hex_to_rgb, poster_meta, render_poster, render_poster_from_json


def test_poster_meta_counts_layers():
    raw = {
        "width": 800,
        "height": 600,
        "layers": [
            {"id": "A", "type": "text", "text": "hi", "x": 10, "y": 10},
            {"id": "B", "type": "rect", "x": 0, "y": 0, "width": 100, "height": 100, "fill": "#fff", "visible": False},
            {"id": "C", "type": "price", "text": "99"},
        ],
    }
    meta = poster_meta(raw)
    assert meta["width"] == 800
    assert meta["height"] == 600
    assert meta["layer_count"] == 3
    assert meta["visible_layers"] == 2
    assert "text" in meta["types"]
    assert "rect" in meta["types"]


def test_poster_template_from_dict_defaults():
    tpl = PosterTemplate.from_dict({"layers": []})
    assert tpl.width == 1080
    assert tpl.height == 1350
    assert tpl.layers == []


def test_render_poster_returns_png_bytes():
    raw = {
        "width": 100,
        "height": 100,
        "layers": [
            {"id": "bg", "type": "rect", "x": 0, "y": 0, "width": 100, "height": 100, "fill": "#ff0000"},
            {"id": "t", "type": "text", "text": "Hi", "x": 10, "y": 10, "fontSize": 20, "color": "#000000"},
        ],
    }
    data = render_poster_from_json(raw)
    assert data[:4] == b"\x89PNG"
    assert len(data) > 100


def test_render_poster_skips_invisible_layers():
    tpl = PosterTemplate.from_dict(
        {
            "width": 50,
            "height": 50,
            "layers": [
                {
                    "id": "x",
                    "type": "rect",
                    "x": 0,
                    "y": 0,
                    "width": 50,
                    "height": 50,
                    "fill": "#00ff00",
                    "visible": False,
                },
            ],
        }
    )
    data = render_poster(tpl)
    assert data[:4] == b"\x89PNG"


def test_hex_to_rgb_3_char():
    assert _hex_to_rgb("#fff") == (255, 255, 255)
    assert _hex_to_rgb("#000") == (0, 0, 0)
    assert _hex_to_rgb("#abc") == (170, 187, 204)
