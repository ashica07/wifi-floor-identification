import yaml
import os
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    root = Path(cfg["paths"]["root_dir"])
    cfg["paths"]["csv_dir"] = root / cfg["paths"]["csv_dir"]
    cfg["paths"]["out_full"] = root / cfg["paths"]["out_full"]
    cfg["paths"]["out_balanced"] = root / cfg["paths"]["out_balanced"]
    
    return cfg