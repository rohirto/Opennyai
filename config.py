from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent


# Input / output
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"


# Supported documents
SUPPORTED_EXTENSIONS = {".pdf"}


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

LLM_PROVIDER = "qwen"       # "qwen" or "gemini"

# Qwen / Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct"
OLLAMA_TIMEOUT = 180
OLLAMA_KEEP_ALIVE = "10m"

# Gemini
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_API_KEY = None
GEMINI_TIMEOUT = 180

# Common generation parameters
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.9
LLM_MAX_OUTPUT_TOKENS = 1200

# Gemini API key
GEMINI_API_KEY = None


def ensure_directories():
    """Create required project directories if they do not exist."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)