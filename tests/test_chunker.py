import pytest

from src.chunking.text_chunker import chunk_pages


def test_chunks_keep_source_metadata_and_split_long_pages_with_overlap():
    pages = [
        {
            "page_number": 4,
            "text": "abcdefghij",
            "char_count": 10,
            "file_name": "guide.pdf",
            "total_pages": 9,
        },
        {
            "page_number": 5,
            "text": "XY",
            "char_count": 2,
            "file_name": "guide.pdf",
            "total_pages": 9,
        },
    ]

    chunks = chunk_pages(pages, max_chars=5, overlap=2)

    assert [chunk["text"] for chunk in chunks] == ["abcde", "defgh", "ghij", "XY"]
    assert [chunk["page_number"] for chunk in chunks] == [4, 4, 4, 5]
    assert [chunk["file_name"] for chunk in chunks] == ["guide.pdf"] * 4
    assert [chunk["total_pages"] for chunk in chunks] == [9] * 4
    assert [chunk["char_count"] for chunk in chunks] == [5, 5, 4, 2]


def test_chunks_skip_empty_pages_without_changing_source_page_numbers():
    pages = [
        {"page_number": 1, "text": "", "char_count": 0, "file_name": "a.pdf", "total_pages": 3},
        {"page_number": 2, "text": "   ", "char_count": 3, "file_name": "a.pdf", "total_pages": 3},
        {"page_number": 3, "text": "kept", "char_count": 4, "file_name": "a.pdf", "total_pages": 3},
    ]

    chunks = chunk_pages(pages, max_chars=10, overlap=0)

    assert [chunk["text"] for chunk in chunks] == ["kept"]
    assert chunks[0]["page_number"] == 3


def test_chunking_is_deterministic_and_empty_input_returns_no_chunks():
    pages = [{"page_number": 1, "text": "abcdefgh", "char_count": 8, "file_name": "a.pdf", "total_pages": 1}]

    first = chunk_pages(pages, max_chars=4, overlap=1)

    assert first == chunk_pages(pages, max_chars=4, overlap=1)
    assert chunk_pages([], max_chars=4, overlap=1) == []


@pytest.mark.parametrize(
    ("max_chars", "overlap"),
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6), (True, 0), (5, 1.5)],
)
def test_chunking_rejects_invalid_size_and_overlap(max_chars, overlap):
    with pytest.raises((TypeError, ValueError)):
        chunk_pages([], max_chars=max_chars, overlap=overlap)
