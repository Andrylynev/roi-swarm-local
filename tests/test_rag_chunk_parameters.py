import pytest

from roi_swarm.rag_cli import _chunks


@pytest.mark.parametrize("chunk_size", [0, -1, True, 2.5, "4", None])
@pytest.mark.parametrize("text", ["abcdefghij", "   "])
def test_chunks_reject_invalid_sizes(chunk_size, text):
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        list(_chunks(text, chunk_size=chunk_size, overlap=0))


@pytest.mark.parametrize("overlap", [-1, 4, 5, True, 1.5, "1", None])
@pytest.mark.parametrize("text", ["abcdefghij", ""])
def test_chunks_reject_invalid_overlaps(overlap, text):
    with pytest.raises(ValueError, match="overlap must be an integer"):
        list(_chunks(text, chunk_size=4, overlap=overlap))


@pytest.mark.parametrize(
    "overlap, expected",
    [
        (0, ["abcd", "efgh", "ij"]),
        (1, ["abcd", "defg", "ghij"]),
        (3, ["abcd", "bcde", "cdef", "defg", "efgh", "fghi", "ghij"]),
    ],
)
def test_chunks_preserve_valid_overlap_behavior(overlap, expected):
    assert list(_chunks("  abcdefghij\n", chunk_size=4, overlap=overlap)) == expected


def test_chunks_support_single_character_chunks_and_empty_input():
    assert list(_chunks("abc", chunk_size=1, overlap=0)) == ["a", "b", "c"]
    assert list(_chunks("   ")) == []


def test_default_chunks_retain_size_overlap_and_final_tail():
    text = "a" * 1800 + "b" * 500
    chunks = list(_chunks(text))
    assert chunks == [text[:1800], text[1600:]]
    assert all(0 < len(chunk) <= 1800 for chunk in chunks)
