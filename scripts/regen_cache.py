"""Regenerate pickle caches for all splits. Run as a script (not as -m)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load import load_targets, load_dep_bodies, load_dep_edges, print_stats


def regen(table: str, cache_dir: str = "cache") -> None:
    print(f"\nRegenerating {table}...")
    Path(cache_dir).mkdir(exist_ok=True)
    for path in Path(cache_dir).glob(f"{table}_*.pkl"):
        path.unlink()
        print(f"  deleted {path}")
    load_targets(table=table, cache_dir=cache_dir)
    load_dep_edges(table=table, cache_dir=cache_dir)
    load_dep_bodies(table=table, cache_dir=cache_dir)
    print_stats(table=table, cache_dir=cache_dir)


if __name__ == "__main__":
    tables = sys.argv[1:] or ["rl_test_100", "rl_train", "rl_val", "rl_test"]
    for t in tables:
        regen(t)
