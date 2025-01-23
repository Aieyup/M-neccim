import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

# .env dosyasını yükle
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Gemini API yapılandırması
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY .env dosyasında bulunamadı")

# Gemini modelini yapılandır
genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel('gemini-pro')

def get_gemini_model():
    """Yapılandırılmış Gemini modelini döndürür."""
    return MODEL 