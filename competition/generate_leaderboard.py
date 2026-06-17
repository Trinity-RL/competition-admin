#!/usr/bin/env python3
"""
Generate the TMRC leaderboard README from per-run JSON files.

Usage:
  python3 generate_leaderboard.py <runs_dir> <readme_path>

Reads all *.json files in runs_dir.
For each participant, keeps their best run per map:
  - Finished runs: lowest race_time_ms wins
  - DNF runs: highest checkpoints_passed (used only if no finished run exists)
Writes the leaderboard between <!-- LEADERBOARD_START --> and <!-- LEADERBOARD_END -->
markers in the README, preserving everything outside those markers.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


LEADERBOARD_START = "<!-- LEADERBOARD_START -->"
LEADERBOARD_END = "<!-- LEADERBOARD_END -->"


def format_time(ms: int | None) -> str:
    if ms is None:
        return "DNF"
    total_s = ms / 1000
    minutes = int(total_s // 60)
    seconds = total_s % 60
    return f"{minutes}:{seconds:06.3f}"


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except Exception as e:
            print(f"WARNING: could not parse {path.name}: {e}", file=sys.stderr)
    return runs


def best_runs_per_participant(runs: list[dict]) -> dict:
    """
    Returns {participant: {map_id: best_result_dict}}.
    Best = finished run with lowest race_time_ms, else DNF with most checkpoints.
    """
    best: dict[str, dict[str, dict]] = {}
    for run in runs:
        participant = run.get("participant", "unknown")
        for map_result in run.get("maps", []):
            map_id = map_result["id"]
            best.setdefault(participant, {})
            prev = best[participant].get(map_id)

            if prev is None:
                best[participant][map_id] = map_result
                continue

            # Finished beats DNF; among finishers lowest time wins; among DNFs most CPs win
            cur_finished = map_result.get("finished", False)
            prev_finished = prev.get("finished", False)

            if cur_finished and not prev_finished:
                best[participant][map_id] = map_result
            elif cur_finished and prev_finished:
                if (map_result["race_time_ms"] or 0) < (prev["race_time_ms"] or 0):
                    best[participant][map_id] = map_result
            elif not cur_finished and not prev_finished:
                if (map_result["checkpoints_passed"] or 0) > (prev["checkpoints_passed"] or 0):
                    best[participant][map_id] = map_result

    return best


def map_order(runs: list[dict]) -> list[tuple[str, str]]:
    """Return [(map_id, map_name), ...] in the order they appear in run files."""
    seen: dict[str, str] = {}
    for run in runs:
        for m in run.get("maps", []):
            if m["id"] not in seen:
                seen[m["id"]] = m["name"]
    return list(seen.items())


def rank_participants(best: dict, map_id: str) -> list[tuple[str, dict]]:
    """Return participants sorted: finishers by time asc, then DNF by checkpoints desc."""
    rows = [
        (participant, result)
        for participant, maps in best.items()
        if (result := maps.get(map_id)) is not None
    ]
    finishers = sorted(
        [(p, r) for p, r in rows if r.get("finished")],
        key=lambda x: x[1]["race_time_ms"] or 0,
    )
    dnf = sorted(
        [(p, r) for p, r in rows if not r.get("finished")],
        key=lambda x: -(x[1]["checkpoints_passed"] or 0),
    )
    return finishers + dnf


def build_leaderboard(runs: list[dict]) -> str:
    if not runs:
        return "_No runs submitted yet._\n"

    best = best_runs_per_participant(runs)
    maps = map_order(runs)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"*Last updated: {now}*\n"]

    for map_id, map_name in maps:
        lines.append(f"\n## {map_name}\n")
        lines.append("| Rank | Participant | Time | Checkpoints | Last CP Time |")
        lines.append("|------|-------------|------|-------------|--------------|")

        ranked = rank_participants(best, map_id)
        if not ranked:
            lines.append("| — | *No submissions yet* | — | — | — |")
            continue

        for rank, (participant, result) in enumerate(ranked, start=1):
            total = result.get("total_checkpoints") or 0
            passed = result.get("checkpoints_passed") or 0
            cp_str = f"{passed}/{total} ✓" if result.get("finished") else f"{passed}/{total}"
            lines.append(
                f"| {rank} "
                f"| [{participant}](https://github.com/{participant}) "
                f"| {format_time(result.get('race_time_ms'))} "
                f"| {cp_str} "
                f"| {format_time(result.get('final_checkpoint_time_ms'))} |"
            )

    return "\n".join(lines) + "\n"


def update_readme(readme_path: Path, leaderboard: str):
    content = readme_path.read_text() if readme_path.exists() else ""

    board_block = f"{LEADERBOARD_START}\n{leaderboard}{LEADERBOARD_END}"

    if LEADERBOARD_START in content:
        start = content.index(LEADERBOARD_START)
        end = content.index(LEADERBOARD_END) + len(LEADERBOARD_END)
        new_content = content[:start] + board_block + content[end:]
    else:
        # Append markers at the end
        separator = "\n\n" if content and not content.endswith("\n\n") else ""
        new_content = content + separator + board_block + "\n"

    readme_path.write_text(new_content)
    print(f"README updated: {readme_path}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <runs_dir> <readme_path>")
        sys.exit(1)

    runs_dir = Path(sys.argv[1])
    readme_path = Path(sys.argv[2])

    runs = load_runs(runs_dir)
    print(f"Loaded {len(runs)} run(s) from {runs_dir}")

    leaderboard = build_leaderboard(runs)
    update_readme(readme_path, leaderboard)


if __name__ == "__main__":
    main()
