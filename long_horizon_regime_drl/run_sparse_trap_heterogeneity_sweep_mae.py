from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import long_horizon_regime_drl.run_sparse_trap_heterogeneity_sweep as sweep


OUT_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"

sweep.SUMMARY_PATH = OUT_DIR / "sparse_trap_heterogeneity_sweep_30seed_mae_summary.csv"
sweep.CURVES_PATH = OUT_DIR / "sparse_trap_heterogeneity_sweep_30seed_mae_curves.csv"
sweep.AGG_PATH = OUT_DIR / "sparse_trap_heterogeneity_sweep_30seed_mae_aggregated.csv"
sweep.TEST_PATH = OUT_DIR / "sparse_trap_heterogeneity_sweep_30seed_mae_paired_tests.csv"
sweep.BOXPLOT_PATH = FIG_DIR / "sparse_trap_heterogeneity_sweep_30seed_mae_boxplots.png"


def main() -> None:
    # Use a separate output path because the older 30-seed run did not record
    # exact advantage error MAE.
    sweep.seed_from_existing_10seed = lambda: None
    sweep.main()


if __name__ == "__main__":
    main()
