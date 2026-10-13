"""Retrieval module for HybridDoc AI."""


def __getattr__(name: str):
    if name in ("search_pdf", "format_search_results", "BM25Searcher"):
        from src.retrieval.search_pdf import BM25Searcher, format_search_results, search_pdf

        mapping = {
            "search_pdf": search_pdf,
            "format_search_results": format_search_results,
            "BM25Searcher": BM25Searcher,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["search_pdf", "format_search_results", "BM25Searcher"]
