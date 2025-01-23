import asyncio
from pathlib import Path
from src.client.agent_client import AgentClient
from src.utils.logger import logger
from src.utils.data_helpers import format_analysis_result
from src.config.settings import OUTPUT_DIR
from src.client.exceptions import AgentClientError
from dotenv import load_dotenv
import locale
import os
import sys

# .env dosyasını yükle
load_dotenv()

# Türkçe karakter desteği için locale ayarı
try:
    locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'tr_TR')
    except:
        logger.warning("Türkçe locale ayarlanamadı, varsayılan kullanılacak")

async def main():
    """CSV analiz örneği."""
    try:
        # Output directory kontrolü
        if not OUTPUT_DIR.exists():
            logger.info(f"Output dizini oluşturuluyor: {OUTPUT_DIR}")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # CSV dosya yolu kontrolü
        csv_path = Path("Sample_01.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")
        
        # İstemciyi oluştur
        client = AgentClient()
        
        # CSV analizi yap
        logger.info(f"CSV analizi başlıyor: {csv_path}")
        analysis_result = await client.analyze_csv(str(csv_path))
        
        if analysis_result["status"] == "error":
            logger.error(f"Analiz hatası: {analysis_result.get('error')}")
            return
        
        # Sonuçları göster
        print("\n=== ÇALIŞAN VERİLERİ ANALİZİ ===")
        
        print("\n1. Temel İstatistikler:")
        basic_stats = analysis_result["results"]["basic_stats"]
        print(f"- Toplam çalışan sayısı: {basic_stats['toplam_calisan']}")
        print(f"- Ortalama maaş: {basic_stats['ortalama_maas']:.2f} TL")
        print(f"- Maaş standart sapması: {basic_stats['maas_std']:.2f} TL")
        
        print("\n2. Departman Analizi:")
        for dept, stats in analysis_result["results"]["department_analysis"].items():
            print(f"\n{dept}:")
            print(f"- Çalışan sayısı: {stats['calisan_sayisi']}")
            print(f"- Ortalama maaş: {stats['ortalama_maas']:.2f} TL")
            print(f"- Maaş aralığı: {stats['min_maas']:.2f} - {stats['max_maas']:.2f} TL")
        
        print("\n3. Yaş Dağılımı:")
        age_dist = analysis_result["results"]["age_distribution"]
        print(f"- Ortalama yaş: {age_dist['ortalama_yas']:.1f}")
        print(f"- En genç: {age_dist['min_yas']:.1f}")
        print(f"- En yaşlı: {age_dist['max_yas']:.1f}")
        print("\nYaş grupları:")
        for grup, sayi in age_dist["yas_dagilimi"].items():
            print(f"- {grup}: {sayi} kişi")
        
        print("\n4. Gemini AI Analizi:")
        print(analysis_result["results"]["gemini_analysis"])
        
        print("\n5. Oluşturulan Görseller:")
        for viz_file in analysis_result["results"]["visualization_files"]:
            print(f"- {viz_file}")
        
        # Sonuçları kaydet
        output_file = OUTPUT_DIR / f"analysis_result_{analysis_result['analysis_id']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(format_analysis_result(analysis_result))
        logger.info(f"Analiz sonuçları kaydedildi: {output_file}")
        
    except AgentClientError as ace:
        logger.error(f"İstemci hatası: {str(ace)}")
        print(f"Hata: {str(ace)}")
    except Exception as e:
        logger.error(f"Program hatası: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 