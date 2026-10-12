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


def test_chunk_formula_pages_handles_formula_group_atomically_when_oversized():
    from src.chunking.text_chunker import chunk_formula_pages
    f1 = {'id': 'p1-f1', 'original_text': 'K =', 'latex': 'K =', 'page_number': 1, 'bbox': [10, 10, 20, 20]}
    f2 = {'id': 'p1-f2', 'original_text': '2^{-2/3}', 'latex': '2^{-2/3}', 'page_number': 1, 'bbox': [10, 30, 20, 40]}
    grp = {
        'id': 'p1-f1+p1-f2',
        'kind': 'wrapped_relation',
        'formula_ids': ['p1-f1', 'p1-f2'],
        'formulas': [f1, f2],
        'latex': 'K = 2^{-2/3}',
        'state': 'unverified',
        'answer_eligible': False,
    }
    block_text = "[UNVERIFIED FORMULA GROUP p1-f1+p1-f2; heuristic wrapped relation; not answer evidence]\n\\[ K = 2^{-2/3} \\]"
    page = {
        'page_number': 1,
        'file_name': 'test.pdf',
        'source_sha256': 'abc',
        'answer_eligible': False,
        'formulas': [f1, f2],
        'formula_groups': [grp],
        'blocks': [
            {'kind': 'text', 'text': 'Short preceding text.'},
            {'kind': 'formula_group', 'group_id': grp['id'], 'formula_ids': grp['formula_ids'], 'group': grp, 'text': block_text},
            {'kind': 'text', 'text': 'Trailing text.'},
        ],
    }
    # max_chars=15 is far smaller than the 110-character formula group block
    chunks = chunk_formula_pages([page], max_chars=15, overlap=2)
    # The oversized formula group block must remain ONE single atomic chunk (not sliced)
    grp_chunks = [c for c in chunks if c.get('formula_groups')]
    assert len(grp_chunks) == 1
    c = grp_chunks[0]
    assert c['text'] == block_text
    assert c['char_count'] == len(block_text)
    assert len(c['formulas']) == 2
    assert [f['id'] for f in c['formulas']] == ['p1-f1', 'p1-f2']
    assert not c['answer_eligible']


def test_chunk_formula_pages_preserves_legacy_inputs_without_groups():
    from src.chunking.text_chunker import chunk_formula_pages
    f1 = {'id': 'p1-f1', 'original_text': 'x=1', 'latex': 'x=1', 'page_number': 1, 'bbox': [0, 0, 1, 1]}
    legacy_page = {
        'page_number': 1,
        'file_name': 'legacy.pdf',
        'source_sha256': 'sha',
        'answer_eligible': False,
        'formulas': [f1],
        'blocks': [
            {'kind': 'text', 'text': 'Legacy intro.'},
            {'kind': 'formula', 'formula_id': 'p1-f1', 'text': '[FORMULA p1-f1]'},
        ],
    }
    chunks = chunk_formula_pages([legacy_page], max_chars=50, overlap=5)
    assert len(chunks) == 2
    assert chunks[1]['formulas'] == [f1]
