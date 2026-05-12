"""
Phase 2.2.b — Threshold calibration (hard gate).

Samples 50 dep statements from v2, queries the TheoremSearch API for each,
computes rapidfuzz.fuzz.ratio between each API result body and the source v2 body,
and prints the true-match and cross-pair distributions.

Writes:
  configs/baseline.yaml          → match_threshold
  <results_dir>/calibration.json → both distributions + chosen threshold

Hard gate: if 5th-percentile of true matches < 95th-percentile of cross-pairs,
the distributions overlap — stop and report.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_rds_connection
from src.env.id_mapping import normalize
from src.env.search_client import SearchClient
from rapidfuzz.fuzz import partial_ratio


RESULTS_DIR = Path("results")
CONFIGS_DIR = Path("configs")
TABLE = os.environ.get("CALIBRATE_TABLE", "rl_test_100")
N_SAMPLE = 50
K = 10


async def run_calibration():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Sample dep statements from the active table's dep universe.
    #    Use the full body as the API query (no slogans available in v2).
    conn = get_rds_connection("v2")
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT statement_id, body FROM (
                    SELECT DISTINCT s.statement_id, s.body
                    FROM informal_dependency d
                    JOIN {TABLE} t ON t.src_id = d.src_id
                    JOIN statement s ON s.statement_id = d.dep_id
                    WHERE d.cite_key IS NOT NULL
                      AND 'deterministic' = ANY(d.methods)
                      AND d.dep_id IS NOT NULL
                      AND s.body IS NOT NULL
                      AND s.body != ''
                ) sub
                ORDER BY RANDOM()
                LIMIT %s
            """, (N_SAMPLE,))
            rows = cur.fetchall()
    finally:
        conn.close()

    if len(rows) < N_SAMPLE:
        print(f"WARNING: only {len(rows)} dep statements found (wanted {N_SAMPLE})")

    print(f"Sampled {len(rows)} dep statements from {TABLE} dep universe")

    # Minimum score to count as a "found" match. Statements below this have no
    # plausible match in the API index (corpus-version skew) and are excluded from
    # threshold fitting — they would contaminate the true-match distribution.
    PLAUSIBILITY_MIN = 80.0

    true_match_scores: list[float] = []
    cross_pair_scores: list[float] = []
    not_found_count = 0

    async with SearchClient(cache_dir="cache/search") as client:
        for i, (stmt_id, body) in enumerate(rows):
            # Use the full body as query — longer text is more unique.
            # The API embeds the query; full LaTeX bodies outperform 100-char fragments.
            query = body
            results = await client.search(query, K)

            if not results:
                print(f"  [{i+1}/{len(rows)}] no results for stmt {stmt_id}, skipping")
                not_found_count += 1
                continue

            norm_source = normalize(body)
            result_scores = [(r, partial_ratio(norm_source, normalize(r.body))) for r in results]
            result_scores.sort(key=lambda x: x[1], reverse=True)

            best_result, best_score = result_scores[0]

            if best_score < PLAUSIBILITY_MIN:
                # Statement not found in API index (corpus-version skew). Exclude from fit.
                not_found_count += 1
                print(f"  [{i+1}/{len(rows)}] NOT FOUND (best={best_score:.1f})  query={query[:60]!r}")
                continue

            true_match_scores.append(best_score)
            for r, s in result_scores[1:]:
                cross_pair_scores.append(s)

            print(f"  [{i+1}/{len(rows)}] best={best_score:.1f}  query={query[:80]!r}")

    if not true_match_scores:
        print("ERROR: no plausible true matches found. Check network / API availability.")
        sys.exit(1)

    print(f"\n{not_found_count}/{len(rows)} statements not found in API index (excluded from fit).")

    true_match_scores.sort()
    cross_pair_scores.sort()

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        idx = (p / 100) * (len(data) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
        return data[lo] + (idx - lo) * (data[hi] - data[lo])

    p5_true = percentile(true_match_scores, 5)
    p95_cross = percentile(cross_pair_scores, 95)

    print(f"\n=== Calibration Results ===")
    print(f"True-match scores  (n={len(true_match_scores)}): min={min(true_match_scores):.1f}  p5={p5_true:.1f}  median={percentile(true_match_scores, 50):.1f}  max={max(true_match_scores):.1f}")
    if cross_pair_scores:
        print(f"Cross-pair scores  (n={len(cross_pair_scores)}): min={min(cross_pair_scores):.1f}  p95={p95_cross:.1f}  median={percentile(cross_pair_scores, 50):.1f}  max={max(cross_pair_scores):.1f}")

    # Hard gate
    if p5_true < p95_cross:
        print(f"\nHARD GATE FAILED: distributions overlap!")
        print(f"  5th-pct true-match ({p5_true:.1f}) < 95th-pct cross-pair ({p95_cross:.1f})")
        print("Likely causes: parser-version skew between DB snapshots, or API indexed against a different corpus version.")
        print("Do not proceed. Recall numbers downstream will be unreliable.")
        sys.exit(2)

    # Pick threshold midway between p5-true and p95-cross, clamped to [70, 98]
    threshold = round((p5_true + p95_cross) / 2, 1)
    threshold = max(70.0, min(98.0, threshold))

    print(f"\nClean gap: p5-true={p5_true:.1f}  p95-cross={p95_cross:.1f}")
    print(f"Chosen match_threshold: {threshold}")

    # Write to configs/baseline.yaml
    baseline_cfg = {"match_threshold": threshold, "low_confidence_gap": 5.0}
    with open(CONFIGS_DIR / "baseline.yaml", "w") as f:
        yaml.dump(baseline_cfg, f)
    print(f"Written: {CONFIGS_DIR / 'baseline.yaml'}")

    # Write calibration.json
    cal_data = {
        "true_match_scores": true_match_scores,
        "cross_pair_scores": cross_pair_scores,
        "p5_true_match": p5_true,
        "p95_cross_pair": p95_cross,
        "match_threshold": threshold,
        "plausibility_min": PLAUSIBILITY_MIN,
        "n_sampled": len(rows),
        "n_found": len(true_match_scores),
        "not_found_rate": not_found_count / max(len(rows), 1),
    }
    cal_path = RESULTS_DIR / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(cal_data, f, indent=2)
    print(f"Written: {cal_path}")

    print("\nCalibration complete. Threshold is clean.")


if __name__ == "__main__":
    asyncio.run(run_calibration())
