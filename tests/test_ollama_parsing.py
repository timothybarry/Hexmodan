from azmo_mind.providers.ollama import _extract_json_object


def test_extract_json_from_markdown_fence():
    raw = '```json\n{"speech":"A","emotion":"neutral"}\n```'
    assert _extract_json_object(raw).startswith('{"speech"')


def test_extract_json_after_small_preamble():
    raw = 'Result follows: {"speech":"A","emotion":"neutral"} trailing'
    assert _extract_json_object(raw) == '{"speech":"A","emotion":"neutral"}'
