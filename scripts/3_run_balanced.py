import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import load_config
from src.data_utils import load_and_process_data
from src.experiments import run_all_experiments

def main():
    cfg = load_config()
    csv_path = Path(cfg["paths"]["csv_dir"]) / cfg["datasets"]["balanced"]
    out_dir = Path(cfg["paths"]["out_balanced"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f" RUNNING BALANCED DATASET: {csv_path}")
    df, W = load_and_process_data(csv_path)
    summary = run_all_experiments(df, W)
    
    out_file = out_dir / "balanced_results.csv"
    summary.to_csv(out_file, index=False)
    print(f"\n Balanced results saved to: {out_file}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()