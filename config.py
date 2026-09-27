"""
Single source of truth for FINDETECT paths, model weights,
thresholds, and runtime settings. Everything here can be
overridden with an environment variable so the pipeline stays
portable across machines and environments (dev/staging/prod)
without editing code.
"""
 
import logging
import os
from pathlib import Path
 
# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
 
FINAL_DATA_DIR = Path(
    os.environ.get(
        "FINDETECT_FINAL_DATA_DIR",
        REPO_ROOT / "Data" / "processed" / "FINDETECT_Block_D_Numerical_Final",
    )
).expanduser()
 
FINAL_GLOB = str(FINAL_DATA_DIR / "block_d_numerical_bucket_*.parquet")
 
OUTPUT_ROOT = Path(
    os.environ.get("FINDETECT_OUTPUT_DIR", REPO_ROOT / "outputs" / "FINDETECT_Pipeline_Output")
).expanduser().resolve()
 
TEMP_ROOT = Path(
    os.environ.get("FINDETECT_TEMP_DIR", REPO_ROOT / ".tmp" / "findetect_daily")
).expanduser().resolve()
 
SAMPLE_INPUT_PATH = Path(
    os.environ.get("FINDETECT_SAMPLE_INPUT", REPO_ROOT / "Data" / "processed" / "block_d_numerical_bucket_000.parquet")
).expanduser()
 
SAMPLE_OUTPUT_PATH = Path(
    os.environ.get("FINDETECT_SAMPLE_OUTPUT", REPO_ROOT / "Data" / "processed" / "sample_1000k.parquet")
).expanduser()
 
XGBOOST_MODEL_PATH = Path(
    os.environ.get("FINDETECT_XGBOOST_MODEL", REPO_ROOT / "models" / "final_xgboost_aml_model.json")
).expanduser()
 
AUTOENCODER_MODEL_DIR = Path(
    os.environ.get("AUTOENCODER_MODEL_DIR", REPO_ROOT / "models" / "autoencoder_final64")
).expanduser().resolve()
 
AUTOENCODER_MODEL_FILE = "autoencoder_final64_fixed.keras"
AUTOENCODER_BUNDLE_FILE = "preprocessing_and_scoring_final64.joblib"
 
DEFAULT_PARQUET_PATH = Path(
    os.environ.get("FINDETECT_DEFAULT_PARQUET", REPO_ROOT / "Data" / "processed" / "prod_sample_oct23_25.parquet")
).expanduser()
 
# ---------------------------------------------------------------
# Risk fusion weights (must sum to 1.0)
# ---------------------------------------------------------------
XGB_WEIGHT = float(os.environ.get("FINDETECT_XGB_WEIGHT", 0.15 ))
ANOMALY_WEIGHT = float(os.environ.get("FINDETECT_ANOMALY_WEIGHT", 0.65))
BEHAVIORAL_WEIGHT = float(os.environ.get("FINDETECT_BEHAVIORAL_WEIGHT", 0.20 ))
 
# ---------------------------------------------------------------
# Behavioral model sub-weights (must sum to 1.0)
# ---------------------------------------------------------------
AMOUNT_DRIFT_WEIGHT = float(os.environ.get("FINDETECT_AMOUNT_DRIFT_WEIGHT", 0.35))
FREQUENCY_DRIFT_WEIGHT = float(os.environ.get("FINDETECT_FREQUENCY_DRIFT_WEIGHT", 0.25))
PEER_GROUP_WEIGHT = float(os.environ.get("FINDETECT_PEER_GROUP_WEIGHT", 0.25))
VELOCITY_WEIGHT = float(os.environ.get("FINDETECT_VELOCITY_WEIGHT", 0.15))
 
MIN_PEER_GROUP_SIZE = int(os.environ.get("FINDETECT_MIN_PEER_GROUP_SIZE", 20))
ROBUST_Z_SATURATION = float(os.environ.get("FINDETECT_ROBUST_Z_SATURATION", 5.0))
MAD_SCALE_FACTOR = 1.4826  # constant: converts MAD to a normal-equivalent std dev
 
# ---------------------------------------------------------------
# Graph / investigation thresholds
# ---------------------------------------------------------------
SEED_THRESHOLD = float(os.environ.get("FINDETECT_SEED_THRESHOLD", 0.45))
GRAPH_RECEIVING_PENALTY = float(os.environ.get("FINDETECT_GRAPH_PENALTY", 0.3046))
GRAPH_INDIRECT_PENALTY = float(os.environ.get("FINDETECT_INDIRECT_PENALTY", 0.0555))  # NEW
INVESTIGATION_THRESHOLD = float(os.environ.get("FINDETECT_INVESTIGATION_THRESHOLD", 0.38))
MEDIUM_RISK_THRESHOLD = float(os.environ.get("FINDETECT_MEDIUM_RISK_THRESHOLD", 0.30))
# ---------------------------------------------------------------
# LLM chat settings (used by main.py's /chat endpoint)
# ---------------------------------------------------------------
OLLAMA_HOST = os.environ.get("FINDETECT_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("FINDETECT_OLLAMA_MODEL", "qwen2.5")
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("FINDETECT_OLLAMA_TIMEOUT", 120.0))
OLLAMA_TEMPERATURE = float(os.environ.get("FINDETECT_OLLAMA_TEMPERATURE", 0.1))
OLLAMA_NUM_PREDICT = int(os.environ.get("FINDETECT_OLLAMA_NUM_PREDICT", 400))
 
# ---------------------------------------------------------------
# API / server settings
# ---------------------------------------------------------------
# Comma-separated list of allowed origins, e.g. "https://app.example.com,https://admin.example.com"
# Defaults to no cross-origin access allowed; set explicitly for your deployment.
_raw_origins = os.environ.get("FINDETECT_CORS_ORIGINS", "")
CORS_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
 
# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
LOG_LEVEL = os.environ.get("FINDETECT_LOG_LEVEL", "INFO").upper()
 
 
def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured once for the whole process."""
    root = logging.getLogger("findetect")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        root.addHandler(handler)
        root.setLevel(LOG_LEVEL)
    return logging.getLogger(f"findetect.{name}")
 