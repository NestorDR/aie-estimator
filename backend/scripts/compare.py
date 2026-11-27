# scripts/compare.py
"""
Standalone CLI comparing cosine similarity between two texts via OpenAI embeddings.

Reuses the configured ``OpenAIEmbedder`` from ``aie_estimator.dependencies`` and
the pure-stdlib ``cosine_similarity`` helper so it runs identically locally
(``uv run python scripts/compare.py``) and inside the production Docker container
(``python scripts/compare.py``) without ``sys.path`` workarounds.
"""

# --- Python Standard Library ---
import argparse
import asyncio
from contextlib import redirect_stdout
import sys

# --- App modules ---
from aie_estimator.dependencies import get_embedder, get_openai_client
from aie_estimator.generation.rag.analysis.similarity import cosine_similarity
from aie_estimator.generation.rag.embedding.embedder import OpenAIEmbedder


async def _compare_texts(embedder: OpenAIEmbedder, first_text: str, second_text: str) -> float:
    """
    Embed both input texts and return their cosine similarity.

    :param embedder: Configured ``OpenAIEmbedder`` instance.
    :param first_text: First input text to embed.
    :param second_text: Second input text to embed.

    :return: Cosine similarity between the two resulting vectors.
    :raises ValueError: If either input text is empty or whitespace-only.
    """
    vector_a = await embedder.embed_one(first_text)
    vector_b = await embedder.embed_one(second_text)
    return cosine_similarity(vector_a, vector_b)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint printing the cosine similarity of two embedded texts to ``stdout``.

    :param argv: Optional CLI argument list (defaults to ``sys.argv[1:]``).

    :return: Exit code ``0`` on success, ``1`` on missing API key or invalid input.
    """
    parser = argparse.ArgumentParser(
        prog="python scripts/compare.py",
        description="Compute cosine similarity between two texts using the configured RAG embedder.",
    )
    # Digit-prefixed flags require explicit `dest` names so parsed attributes remain valid Python identifiers.
    parser.add_argument("--1st", dest="first_text", required=True, help="First text to compare.")
    parser.add_argument("--2nd", dest="second_text", required=True, help="Second text to compare.")
    args = parser.parse_args(argv)

    # Route early structlog bootstrap lines emitted during Settings load to stderr so stdout stays clean.
    with redirect_stdout(sys.stderr):
        embedder = get_embedder(get_openai_client())

    if embedder is None:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    # SelectorEventLoop avoids ProactorEventLoop socket/transport issues on Windows CLI exit.
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    try:
        similarity = asyncio.run(
            _compare_texts(embedder, args.first_text, args.second_text),
            loop_factory=loop_factory,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Text 1: {args.first_text}")
    print(f"Text 2: {args.second_text}")
    print(f"Cosine similarity: {similarity:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
