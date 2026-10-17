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
# Risk fusion weights 
# ---------------------------------------------------------------
XGB_WEIGHT = float(os.environ.get("FINDETECT_XGB_WEIGHT", 0.15 ))
ANOMALY_WEIGHT = float(os.environ.get("FINDETECT_ANOMALY_WEIGHT", 0.50))
BEHAVIORAL_WEIGHT = float(os.environ.get("FINDETECT_BEHAVIORAL_WEIGHT", 0.35 ))
 
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
SEED_THRESHOLD = float(os.environ.get("FINDETECT_SEED_THRESHOLD", 0.60))
GRAPH_RECEIVING_PENALTY = float(os.environ.get("FINDETECT_GRAPH_PENALTY", 0.3046))
GRAPH_INDIRECT_PENALTY = float(os.environ.get("FINDETECT_INDIRECT_PENALTY", 0.0555))  # NEW
TERTIARY_PENALTY_FACTOR = float(os.environ.get("FINDETECT_TERTIARY_PENALTY_FACTOR", 0.5))
BIDIR_PENALTY = float(os.environ.get("FINDETECT_BIDIR_PENALTY", 0.08))
FAN_BOOST = float(os.environ.get("FINDETECT_FAN_BOOST", 0.12))
GRAPH_PENALTY_CAP = float(os.environ.get("FINDETECT_GRAPH_PENALTY_CAP", 0.75))
INVESTIGATION_THRESHOLD = float(os.environ.get("FINDETECT_INVESTIGATION_THRESHOLD", 0.411))  # 0.522 very good
MEDIUM_RISK_THRESHOLD = float(os.environ.get("FINDETECT_MEDIUM_RISK_THRESHOLD", 0.350))
TIER_2_THRESHOLD = float(os.environ.get("FINDETECT_TIER_2_THRESHOLD", 0.650))
TIER_1_THRESHOLD = float(os.environ.get("FINDETECT_TIER_1_THRESHOLD", 0.522))
# ---------------------------------------------------------------
# LLM chat settings (used by main.py's /chat endpoint)
# ---------------------------------------------------------------
OLLAMA_HOST = os.environ.get("FINDETECT_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("FINDETECT_OLLAMA_MODEL", "qwen2.5")
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("FINDETECT_OLLAMA_TIMEOUT", 120.0))
OLLAMA_TEMPERATURE = float(os.environ.get("FINDETECT_OLLAMA_TEMPERATURE", 0.1))
OLLAMA_NUM_PREDICT = int(os.environ.get("FINDETECT_OLLAMA_NUM_PREDICT", 400))

# ---------------------------------------------------------------
# Agentic investigator settings (agent.py / tools.py)
# ---------------------------------------------------------------
# Master switch for tool-calling. Set FINDETECT_AGENT_ENABLED=0 to fall
# back to the original single-shot chat behaviour.
AGENT_ENABLED = os.environ.get("FINDETECT_AGENT_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
# Model used for the agent loop. Defaults to the chat model, but a larger
# model (e.g. qwen2.5:7b-instruct) is more reliable at tool calling.
OLLAMA_TOOL_MODEL = os.environ.get("FINDETECT_OLLAMA_TOOL_MODEL", OLLAMA_CHAT_MODEL)
# Context window for agent calls. Ollama's qwen2.5 default is only 4096
# tokens, which is too small for system prompt + telemetry + tool results.
AGENT_NUM_CTX = int(os.environ.get("FINDETECT_AGENT_NUM_CTX", 8192))
# Agent calls can be slower than plain chat on CPU-only machines, so they get
# their own timeout. Keep the model resident between turns.
AGENT_OLLAMA_TIMEOUT = float(
    os.environ.get("FINDETECT_AGENT_OLLAMA_TIMEOUT", 300.0)
)
AGENT_KEEP_ALIVE = os.environ.get("FINDETECT_AGENT_KEEP_ALIVE", "30m")
# Load the agent model into Ollama in the background when the API starts.
AGENT_WARMUP = os.environ.get("FINDETECT_AGENT_WARMUP", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
# Maximum tool-calling rounds before the agent is forced to answer.
AGENT_MAX_STEPS = int(os.environ.get("FINDETECT_AGENT_MAX_STEPS", 5))
# Use Ollama native tool calling. Disable on slow CPU-only machines to cut the
# prompt/tool-schema overhead (the JSON protocol and prefetch still work).
AGENT_NATIVE_TOOLS = os.environ.get(
    "FINDETECT_AGENT_NATIVE_TOOLS", "1"
).strip().lower() not in {"0", "false", "no", "off"}
# Conversation turns forwarded back to the model (user + assistant messages).
AGENT_MAX_HISTORY_MESSAGES = int(os.environ.get("FINDETECT_AGENT_MAX_HISTORY", 10))
# Default / maximum number of transaction rows a single tool call may return.
AGENT_TOOL_TX_LIMIT = int(os.environ.get("FINDETECT_AGENT_TX_LIMIT", 25))
AGENT_TOOL_TX_MAX_LIMIT = int(os.environ.get("FINDETECT_AGENT_TX_MAX_LIMIT", 100))

# ---------------------------------------------------------------
# Investigator workflow (cases, notes, SAR, audit)
# ---------------------------------------------------------------
# SQLite database for mutable workflow state. The source risk/transaction
# parquet files remain the source of truth; this database never stores scores
# or transactions.
WORKFLOW_DB = Path(
    os.environ.get(
        "FINDETECT_WORKFLOW_DB",
        REPO_ROOT / "outputs" / "workflow" / "workflow.db",
    )
).expanduser().resolve()

# Pipeline input files must resolve inside this directory. This blocks
# arbitrary path reads through /pipeline/analyze.
DATA_ROOT = Path(
    os.environ.get("FINDETECT_DATA_ROOT", REPO_ROOT / "Data" / "processed")
).expanduser().resolve()

# Optional shared-secret API gate. Empty means local-only mode (no auth).
API_TOKEN = os.environ.get("FINDETECT_API_TOKEN", "").strip()
INVESTIGATOR_ID = os.environ.get("FINDETECT_INVESTIGATOR_ID", "local-investigator").strip()

CASE_PAGE_SIZE_DEFAULT = int(os.environ.get("FINDETECT_CASE_PAGE_SIZE", 25))
CASE_PAGE_SIZE_MAX = int(os.environ.get("FINDETECT_CASE_PAGE_SIZE_MAX", 100))

# Bounded graph exploration. The sample has p99 degree 5 but a few hub
# accounts with tens of thousands of neighbours; these limits protect the API.
GRAPH_MAX_NODES = int(os.environ.get("FINDETECT_GRAPH_MAX_NODES", 100))
GRAPH_MAX_EDGES = int(os.environ.get("FINDETECT_GRAPH_MAX_EDGES", 300))
GRAPH_MAX_DEPTH = int(os.environ.get("FINDETECT_GRAPH_MAX_DEPTH", 2))
GRAPH_MAX_PATHS = int(os.environ.get("FINDETECT_GRAPH_MAX_PATHS", 20))
GRAPH_MAX_CYCLES = int(os.environ.get("FINDETECT_GRAPH_MAX_CYCLES", 20))
GRAPH_MAX_VISITED = int(os.environ.get("FINDETECT_GRAPH_MAX_VISITED", 1000))
GRAPH_MAX_WINDOW_DAYS = int(os.environ.get("FINDETECT_GRAPH_MAX_WINDOW_DAYS", 7))
GRAPH_MAX_PATH_DEPTH = int(os.environ.get("FINDETECT_GRAPH_MAX_PATH_DEPTH", 4))

SAR_MAX_EVIDENCE_ROWS = int(os.environ.get("FINDETECT_SAR_MAX_EVIDENCE_ROWS", 50))

# ---------------------------------------------------------------
# API / server settings
# ---------------------------------------------------------------
# Comma-separated list of allowed origins. Defaults to the local dashboard
# origins; set FINDETECT_CORS_ORIGINS for a networked deployment.
_raw_origins = os.environ.get(
    "FINDETECT_CORS_ORIGINS",
    "http://127.0.0.1:5600,http://localhost:5600,http://127.0.0.1:8000,http://localhost:8000",
)
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
 