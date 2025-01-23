from pathlib import Path

# API ayarları
DEFAULT_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1

# Çıktı dizini ayarları
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"  # Proje kök dizininde output klasörü

# Veri analizi ayarları
DEFAULT_ANALYSIS_METHOD = "static" 