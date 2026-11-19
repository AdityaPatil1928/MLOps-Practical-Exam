import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from feast import FeatureStore

def materialize():
    repo_path = Path(__file__).resolve().parent
    print(f"=== [Feast Materialization] Materializing features into SQLite Online Store ===")
    print(f"Repository path: {repo_path}")
    
    store = FeatureStore(repo_path=str(repo_path))
    end_date = datetime.now(timezone.utc) + timedelta(minutes=5)
    start_date = end_date - timedelta(days=2)
    
    print(f"Materialization Window: [{start_date.isoformat()} -> {end_date.isoformat()}]")
    store.materialize(start_date=start_date, end_date=end_date)
    
    sqlite_db = repo_path / "data" / "online_store.db"
    if sqlite_db.exists():
        db_size = sqlite_db.stat().st_size
        print(f"[SUCCESS] Features materialized successfully into SQLite online store.")
        print(f"Database location: {sqlite_db} ({db_size:,} bytes)")
    else:
        print("[WARNING] SQLite online store file not found after materialization.")

if __name__ == "__main__":
    materialize()
