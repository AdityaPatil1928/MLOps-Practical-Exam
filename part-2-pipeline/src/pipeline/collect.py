import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dataset_loader import get_dataset, DATASET_NAME

def collect(output_path: str):
    print(f"=== [Stage 1: Collect] Ingesting dataset '{DATASET_NAME}' ===")
    df, feature_cols, target_col = get_dataset()
    
    # Add collection timestamp in UTC ISO format
    df["collected_at"] = datetime.now(timezone.utc).isoformat()
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    
    print(f"Dataset successfully collected and saved to: {out_file}")
    print(f"Rows collected: {len(df)}")
    print(f"Columns: {len(df.columns)} ({len(feature_cols)} features, target '{target_col}', collected_at)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Data Collection")
    parser.add_argument("--output", type=str, default="data/raw/raw_data.csv", help="Output raw CSV path")
    args = parser.parse_args()
    collect(args.output)
