"""Read the assignment's JSONL corpus format."""
import json
from typing import List, Tuple


def load_corpus(path: str) -> List[Tuple[str, str]]:
    """Read a corpus.jsonl file (one {"doc_id": ..., "text": ...} object
    per line) and return a list of (doc_id, text) pairs in file order."""
    docs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append((obj["doc_id"], obj["text"]))
    return docs
