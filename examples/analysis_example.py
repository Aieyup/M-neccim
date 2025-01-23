import asyncio
import numpy as np
from pathlib import Path
from src.client.agent_client import AgentClient
from src.utils.logger import logger
from src.utils.data_helpers import prepare_clusters, format_analysis_result
from src.config.settings import OUTPUT_DIR
from src.client.exceptions import AgentClientError
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

async def main():
    """Ana örnek fonksiyonu."""
    # Output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # İstemciyi oluştur
    client = AgentClient()
    
    try:
        # Örnek veri oluştur
        logger.info("Örnek veri oluşturuluyor")
        data = prepare_clusters(n_samples=1000, n_clusters=2)
        
        # Veriyi analiz et
        logger.info("Veri analizi başlıyor")
        analysis_result = await client.analyze_data(data)
        
        if analysis_result["status"] == "error":
            logger.error(f"Analiz hatası: {analysis_result.get('error')}")
            return
            
        print("\nAnaliz Sonucu:")
        print(format_analysis_result(analysis_result))
        
        # Gemini analiz sonuçlarını göster
        if "gemini_analysis" in analysis_result.get("results", {}):
            print("\nGemini Analizi:")
            print(analysis_result["results"]["gemini_analysis"])
        
    except AgentClientError as ace:
        logger.error(f"İstemci hatası: {str(ace)}")
        print(f"Hata: {str(ace)}")
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {str(e)}", exc_info=True)
        print(f"Beklenmeyen hata: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 