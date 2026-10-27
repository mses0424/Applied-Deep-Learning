"""Verify starter hashes, student-block boundaries, and syntax without model execution."""
import ast
from common import ROOT, read_json, sha256

EXPECTED_PREFIX = '''#!/usr/bin/env python3
"""Student-editable retrieval and reranking implementation."""

from __future__ import annotations

from typing import Any


DEFAULT_TOP_K = 10


def retrieve_and_rerank(
    queries: list[dict[str, Any]],
    passages: dict[str, str],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Return one final ranked PID list per query."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(passages) < top_k:
        raise ValueError(
            f"product collection has {len(passages)} items, fewer than top_k={top_k}"
        )

'''


def main():
    for filename, expected in read_json(ROOT / "configs/starter_sha256.json").items():
        if sha256(ROOT / filename) != expected:
            raise SystemExit(f"Starter file changed: {filename}")
    source = (ROOT / "retrieval_utils.py").read_text()
    before, body = source.split("    # ==================== STUDENT CODE START ====================", 1)
    body, after = body.split("    # ===================== STUDENT CODE END =====================", 1)
    if before != EXPECTED_PREFIX or after != "\n\n    return results\n":
        raise SystemExit("Change outside the allowed student block")
    for path in list(ROOT.glob("*.py")) + list((ROOT / "code" / "script").glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    print("Starter hashes, allowed edit boundaries, and Python syntax: PASS")
    print("This verifies code boundaries, not every possible course policy or runtime behavior.")


if __name__ == "__main__":
    main()
