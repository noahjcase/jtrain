"""Enforce per-area coverage floors (TESTING §2) from coverage.py's JSON report.

uv run pytest --cov=jtrainer --cov-branch --cov-report=json
uv run python scripts/check_coverage.py coverage.json
"""

from __future__ import annotations

import json
import sys

# area -> (path prefixes under src/jtrainer/, floor %)
FLOORS = {
    "ingest": (["ingest/"], 90),
    "grading": (["grading.py", "text.py"], 90),
    "model": (["model/"], 90),
    "sampling": (["sampling.py"], 90),
    "game rules": (["game.py"], 90),
    "tagging": (["tagging/"], 80),
    "db": (["db.py"], 80),
}


def main(path: str) -> int:
    with open(path) as fh:
        files = json.load(fh)["files"]
    failed = False
    for area, (prefixes, floor) in FLOORS.items():
        covered = total = 0
        for name, data in files.items():
            rel = name.replace("\\", "/").split("src/jtrainer/")[-1]
            if any(rel.startswith(p) for p in prefixes):
                s = data["summary"]
                covered += s["covered_lines"] + s.get("covered_branches", 0)
                total += s["num_statements"] + s.get("num_branches", 0)
        pct = 100.0 * covered / total if total else 100.0
        status = "ok" if pct >= floor else "BELOW FLOOR"
        failed |= pct < floor
        print(f"{area:<11} {pct:5.1f}%  (floor {floor}%)  {status}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "coverage.json"))
