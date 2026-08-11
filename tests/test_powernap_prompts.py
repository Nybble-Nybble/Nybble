import pytest
from PIL import Image

from nybble.powernap.prompts import (
    actions_xml,
    append_actions,
    append_revise,
    build_think_messages,
    validate_image_paths,
)


def test_three_phase_prompt_keeps_context_stable():
    context = "[12:00:00] [glasses] opens the refrigerator"
    think = build_think_messages(context)
    revise = append_revise(think, "<rationale>food preparation</rationale>", "past lunch")
    actions = append_actions(revise, "<revise>likely cooking</revise>", 3)

    assert think[1]["content"].count(context) == 1
    assert revise[:2] == think
    assert "past lunch" in revise[-1]["content"]
    assert "exactly 3" in actions[-1]["content"]


def test_multimodal_think_prompt_keeps_ordered_past_images():
    think = build_think_messages("context", ("/frames/older.jpg", "/frames/newer.png"))

    assert think[1]["content"] == [
        {
            "type": "text",
            "text": think[1]["content"][0]["text"],
        },
        {"type": "image", "image": "/frames/older.jpg"},
        {"type": "image", "image": "/frames/newer.png"},
    ]
    assert "context" in think[1]["content"][0]["text"]


def test_inkling_image_validation_accepts_png_and_rejects_other_formats(tmp_path):
    png = tmp_path / "frame.png"
    webp = tmp_path / "frame.webp"
    Image.new("RGB", (8, 8)).save(png)
    Image.new("RGB", (8, 8)).save(webp)

    assert validate_image_paths((str(png),)) == (str(png.resolve()),)
    with pytest.raises(ValueError, match="PNG or JPEG"):
        validate_image_paths((str(webp),))


def test_actions_xml_escapes_model_data():
    assert actions_xml(("press <A&B>",)) == (
        "<actions>\n  <action>press &lt;A&amp;B&gt;</action>\n</actions>"
    )
