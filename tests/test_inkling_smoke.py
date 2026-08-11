import pytest
from PIL import Image

pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")
pytest.importorskip("tml_renderers")

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from nybble.powernap.prompts import build_think_messages


def test_inkling_small_tml_v0_renders_real_multimodal_prompt(tmp_path):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (16, 12), (20, 30, 40)).save(frame)
    model = "thinkingmachines/Inkling-Small"
    renderer = renderers.get_renderer("tml_v0", get_tokenizer(model), model_name=model)

    prompt = renderer.build_generation_prompt(
        build_think_messages("<observed_actions />", (str(frame),)), effort=0.0
    )

    assert any(type(chunk).__name__ == "ImageChunk" for chunk in prompt.chunks)
    assert renderer.get_stop_sequences() == [200006]
