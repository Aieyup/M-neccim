import pytest
import numpy as np
from src.client.agent_client import AgentClient
from src.utils.data_helpers import prepare_clusters
from src.config.settings import OUTPUT_DIR

pytestmark = pytest.mark.asyncio

@pytest.mark.integration
class TestIntegration:
    """Entegrasyon testleri."""
    
    async def test_full_analysis_workflow(self):
        """Tam analiz iş akışı testi."""
        client = AgentClient()
        data = prepare_clusters(1000, 3, seed=42)
        
        # Analiz başlatma
        analysis_result = await client.analyze_data(data)
        assert analysis_result["status"] == "success"
        assert "analysis_id" in analysis_result
        
        # Sonuçları alma
        analysis_id = analysis_result["analysis_id"]
        final_result = await client.get_analysis(analysis_id)
        assert final_result["status"] == "success"
        assert "results" in final_result
        
        # Çıktı dizini kontrolü
        assert OUTPUT_DIR.exists() 