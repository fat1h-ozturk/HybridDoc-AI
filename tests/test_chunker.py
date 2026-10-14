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


def test_default_compatibility_is_unchanged():
    pages = [
        {"page_number": 1, "text": "alpha beta gamma delta epsilon", "char_count": 30, "file_name": "t.pdf", "total_pages": 1}
    ]
    # Default boundary should be character mode
    default_chunks = chunk_pages(pages, max_chars=12, overlap=4)
    char_chunks = chunk_pages(pages, max_chars=12, overlap=4, boundary="character")
    assert default_chunks == char_chunks
    # Character mode cuts words (e.g. 'gamma' sliced into 'g')
    assert any("g" in c["text"] and "gamma" not in c["text"] for c in default_chunks)


def test_validation_rejects_invalid_boundary_and_supports_chunk_size_alias():
    pages = [{"page_number": 1, "text": "hello world", "char_count": 11, "file_name": "t.pdf", "total_pages": 1}]
    with pytest.raises(ValueError, match="boundary must be 'character' or 'word'"):
        chunk_pages(pages, boundary="invalid")
    with pytest.raises(ValueError, match="boundary must be 'character' or 'word'"):
        chunk_pages(pages, boundary=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boundary must be 'character' or 'word'"):
        chunk_pages(pages, boundary=123)  # type: ignore[arg-type]

    # chunk_size alias
    chunks_alias = chunk_pages(pages, chunk_size=6, overlap=2, boundary="word")
    chunks_direct = chunk_pages(pages, max_chars=6, overlap=2, boundary="word")
    assert chunks_alias == chunks_direct


def test_no_ordinary_word_cuts_at_end_and_start_boundaries():
    text = "Ayrılabilir diferansiyel denklemler birinci mertebeden diferansiyel denklemlerin en temel türüdür."
    pages = [{"page_number": 1, "text": text, "char_count": len(text), "file_name": "math.pdf", "total_pages": 1}]

    # In character mode, words are cut
    char_chunks = chunk_pages(pages, max_chars=30, overlap=10, boundary="character")
    assert any("denkl" in c["text"] and "denklemler" not in c["text"] for c in char_chunks)

    # In word mode, ordinary whitespace-separated words are NOT cut
    word_chunks = chunk_pages(pages, max_chars=30, overlap=10, boundary="word")
    words_in_text = {w.strip(".,;:?!") for w in text.split()}

    for chunk in word_chunks:
        # Check that chunk length strictly adheres to character budget
        assert len(chunk["text"]) <= 30
        assert chunk["char_count"] == len(chunk["text"])
        # Verify that start and end of chunk text do not cut words
        chunk_words = chunk["text"].strip().split()
        assert len(chunk_words) > 0
        first_word = chunk_words[0].strip(".,;:?!")
        last_word = chunk_words[-1].strip(".,;:?!")
        assert first_word in words_in_text
        assert last_word in words_in_text


def test_coverage_and_reconstruction_zero_overlap():
    text = (
        "İlk bölüm diferansiyel denklemlerin tanımını içerir.\n\n"
        "İkinci bölüm ise ayrılabilir denklemleri ve çözüm adımlarını örneklerle açıklar.\n"
        "Son bölüm alıştırmalar ve kaynakçadan oluşmaktadır."
    )
    pages = [{"page_number": 2, "text": text, "char_count": len(text), "file_name": "book.pdf", "total_pages": 5}]

    chunks = chunk_pages(pages, max_chars=40, overlap=0, boundary="word")
    assert len(chunks) > 1

    # Reconstruction: concatenating chunks must reconstruct the exact original text without loss
    reconstructed = "".join(c["text"] for c in chunks)
    assert reconstructed == text

    # No chunk should exceed the budget
    assert all(c["char_count"] <= 40 for c in chunks)
    assert all(c["char_count"] == len(c["text"]) for c in chunks)


def test_coverage_and_alignment_with_positive_overlap():
    text = "apple banana cherry date elderberry fig grape honeydew kiwi lemon mango nectarine orange"
    pages = [{"page_number": 1, "text": text, "char_count": len(text), "file_name": "fruits.pdf", "total_pages": 1}]

    chunks = chunk_pages(pages, max_chars=25, overlap=10, boundary="word")
    assert len(chunks) > 1

    # Verify all words from the source text are covered in at least one chunk
    all_words = set(text.split())
    covered_words = set()
    for c in chunks:
        assert len(c["text"]) <= 25
        covered_words.update(c["text"].split())
    assert covered_words == all_words

    # Verify that overlap between consecutive chunks aligns to word boundaries
    for i in range(len(chunks) - 1):
        c1 = chunks[i]["text"]
        c2 = chunks[i + 1]["text"]
        words_c1 = c1.split()
        words_c2 = c2.split()
        # The first word of c2 must appear in c1 (sharing overlap context)
        assert words_c2[0] in words_c1


def test_overlap_alignment_progress_and_termination_large_overlap():
    text = "one two three four five six seven eight nine ten"
    pages = [{"page_number": 1, "text": text, "char_count": len(text), "file_name": "t.pdf", "total_pages": 1}]

    # Aggressive overlap: max_chars=10, overlap=9
    chunks = chunk_pages(pages, max_chars=10, overlap=9, boundary="word")
    assert len(chunks) > 0
    assert all(c["char_count"] <= 10 for c in chunks)

    # Reconstruction with overlap=0 on same text
    zero_chunks = chunk_pages(pages, max_chars=10, overlap=0, boundary="word")
    assert "".join(c["text"] for c in zero_chunks) == text


def test_oversized_unbroken_words_bounded_fallback():
    # Long unbroken string exceeding max_chars
    long_token = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
    text = f"prefix {long_token} suffix"
    pages = [{"page_number": 1, "text": text, "char_count": len(text), "file_name": "long.pdf", "total_pages": 1}]

    # Budget is 15 chars, far smaller than the 62-char unbroken token
    chunks = chunk_pages(pages, max_chars=15, overlap=4, boundary="word")

    # Hard budget guarantee: no chunk may ever exceed max_chars
    assert all(len(c["text"]) <= 15 for c in chunks)
    assert all(c["char_count"] <= 15 for c in chunks)

    # Zero overlap must fully retain all characters without dropping anything
    zero_chunks = chunk_pages(pages, max_chars=15, overlap=0, boundary="word")
    assert "".join(c["text"] for c in zero_chunks) == text


def test_unicode_and_newlines_preserved():
    text = "Türkçe karakterler: ç, ğ, ı, ö, ş, ü; büyükler: Ç, Ğ, İ, Ö, Ş, Ü.\n\nMatematik: ∑_{i=1}^n x_i = ∫ f(t) dt\tve limitler."
    pages = [{"page_number": 3, "text": text, "char_count": len(text), "file_name": "uni.pdf", "total_pages": 3}]

    chunks = chunk_pages(pages, max_chars=35, overlap=10, boundary="word")
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 35 for c in chunks)

    # Exact reconstruction on zero overlap preserving newlines and tabs
    zero_chunks = chunk_pages(pages, max_chars=35, overlap=0, boundary="word")
    assert "".join(c["text"] for c in zero_chunks) == text


def test_page_metadata_and_multi_page_boundaries():
    pages = [
        {"page_number": 1, "text": "Page one text content here.", "file_name": "multi.pdf", "total_pages": 2},
        {"page_number": 2, "text": "Page two text content follows.", "file_name": "multi.pdf", "total_pages": 2},
    ]
    chunks = chunk_pages(pages, max_chars=15, overlap=3, boundary="word")

    # Chunks must preserve metadata and not leak across page numbers
    p1_chunks = [c for c in chunks if c["page_number"] == 1]
    p2_chunks = [c for c in chunks if c["page_number"] == 2]

    assert len(p1_chunks) > 1
    assert len(p2_chunks) > 1
    assert len(p1_chunks) + len(p2_chunks) == len(chunks)

    assert all(c["file_name"] == "multi.pdf" for c in chunks)
    assert all(c["total_pages"] == 2 for c in chunks)
    assert all("Page two" not in c["text"] for c in p1_chunks)
    assert all("Page one" not in c["text"] for c in p2_chunks)


def test_chunk_formula_pages_with_word_boundary():
    from src.chunking.text_chunker import chunk_formula_pages
    f1 = {'id': 'p1-f1', 'original_text': 'y = 2x', 'latex': 'y = 2x', 'page_number': 1, 'bbox': [10, 10, 20, 20]}
    page = {
        'page_number': 1,
        'file_name': 'formula_word.pdf',
        'source_sha256': 'sha123',
        'answer_eligible': False,
        'formulas': [f1],
        'blocks': [
            {'kind': 'text', 'text': 'Short introductory explanation of function.'},
            {'kind': 'formula', 'formula_id': 'p1-f1', 'text': '[FORMULA p1-f1]\n\\[ y = 2x \\]'},
            {'kind': 'text', 'text': 'Subsequent prose analyzing behavior at infinity.'},
        ],
    }

    chunks = chunk_formula_pages([page], max_chars=25, overlap=5, boundary="word")
    math_chunks = [c for c in chunks if c.get('formulas')]
    text_chunks = [c for c in chunks if not c.get('formulas')]

    # Formula block remains atomic
    assert len(math_chunks) == 1
    assert math_chunks[0]['text'] == '[FORMULA p1-f1]\n\\[ y = 2x \\]'

    # Text blocks are chunked with word boundaries
    for c in text_chunks:
        assert len(c['text']) <= 25
        # Words must not be sliced
        assert not c['text'].startswith('xplanation')


def test_cli_execution_with_boundary_modes(tmp_path):
    from src.chunking.text_chunker import main

    pdf_file = tmp_path / "dummy.pdf"
    # Test error handling when file missing
    with pytest.raises(SystemExit) as exc:
        main([str(pdf_file)])
    assert exc.value.code == 1

    # Real public fixture execution
    fixture_path = "sample_docs/Fatih_RAG_OCR_Multimodal_PDF_Use_Case.pdf"
    out_file = tmp_path / "chunks.txt"

    code = main([fixture_path, "--boundary", "word", "--chunk-size", "500", "--overlap", "50", "--output", str(out_file)])
    assert code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Boundary Mode: word" in content
    assert "Total Chunks:" in content
    assert "=== Chunk 1" in content
