from nybble.powernap.rendering import decode_model_turn, parse_tagged_block


class Termination:
    def __init__(self, clean):
        self.is_clean = clean


class Renderer:
    def __init__(self, clean=True):
        self.clean = clean
        self.calls = 0

    def parse_response(self, action):
        self.calls += 1
        return {"content": action}, Termination(self.clean)


def text_content(message):
    return message["content"]


def test_length_stop_is_rejected_without_parsing():
    renderer = Renderer()
    decoded = decode_model_turn(
        renderer,
        "partial",
        {"stop_reason": "length"},
        text_content,
    )

    assert not decoded.valid
    assert decoded.error == "max_tokens"
    assert renderer.calls == 0


def test_malformed_or_empty_renderer_turn_is_rejected():
    malformed = decode_model_turn(Renderer(clean=False), "bad", None, text_content)
    empty = decode_model_turn(Renderer(), "  ", None, text_content)

    assert malformed.error == "parse_error"
    assert empty.error == "empty_turn"


def test_clean_renderer_turn_is_stripped_and_accepted():
    decoded = decode_model_turn(Renderer(), "  usable turn  ", None, text_content)

    assert decoded.valid
    assert decoded.text == "usable turn"


def test_phase_tags_require_one_exact_plain_text_root():
    valid = parse_tagged_block("<rationale>short pattern</rationale>", "rationale")
    wrong = parse_tagged_block("<revise>wrong phase</revise>", "rationale")
    nested = parse_tagged_block("<rationale><rationale>nested</rationale></rationale>", "rationale")
    special_token = parse_tagged_block("<|content_model_end_sampling|>", "rationale")

    assert valid.valid and valid.content == "short pattern"
    assert not wrong.valid
    assert not nested.valid
    assert not special_token.valid
