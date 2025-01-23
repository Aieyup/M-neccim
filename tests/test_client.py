import pytest
import numpy as np
from unittest.mock import Mock
from src.client.agent_client import AgentClient
from src.client.exceptions import ValidationError, APIConnectionError, AnalysisError
from src.utils.data_helpers import prepare_clusters
from src.config.settings import DEFAULT_BASE_URL, API_TIMEOUT
from src.config.api_config import get_gemini_model

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_gemini_response():
    class MockResponse:
        text = "Test analiz sonucu"
    return MockResponse()

@pytest.fixture
def client(monkeypatch):
    # Gemini model mock
    mock_model = Mock()
    mock_model.generate_content.return_value = Mock(text="Test analiz sonucu")
    monkeypatch.setattr("src.config.api_config.get_gemini_model", lambda: mock_model)
    return AgentClient()

@pytest.fixture
def sample_data():
    return prepare_clusters(100, 2)

@pytest.fixture
def mock_success_response():
    return {
        "status": "success",
        "analysis_id": "test-123",
        "results": {
            "clusters": 2,
            "metrics": {
                "silhouette_score": 0.8
            }
        }
    }

class TestAgentClient:
    """AgentClient test sınıfı."""
    
    async def test_analyze_data_success(self, client, sample_data):
        result = await client.analyze_data(sample_data)
        assert result["status"] == "success"
        assert "analysis_id" in result
        assert "gemini_analysis" in result["results"]
    
    async def test_analyze_data_with_gemini(self, client, sample_data):
        result = await client.analyze_data(sample_data)
        assert "gemini_analysis" in result["results"]
        assert isinstance(result["results"]["gemini_analysis"], str)
    
    async def test_analyze_data_invalid_input(self, client):
        """Geçersiz veri girişi testi."""
        invalid_inputs = [
            ([1, 2, 3], "numpy array formatında olmalıdır"),
            (np.array([]), "Veri boş olamaz"),
            (np.array([np.nan, 1, 2]), "sonsuz veya NaN değerler var")
        ]
        
        for invalid_input, expected_error in invalid_inputs:
            with pytest.raises(ValidationError) as exc_info:
                await client.analyze_data(invalid_input)
            assert expected_error in str(exc_info.value)
    
    async def test_analyze_data_api_timeout(self, client, sample_data, mock_api):
        """API zaman aşımı testi."""
        mock_api.post(
            f"{DEFAULT_BASE_URL}/analyze",
            exception=TimeoutError("Connection timeout")
        )
        
        with pytest.raises(APIConnectionError) as exc_info:
            await client.analyze_data(sample_data)
        assert "zaman aşımına uğradı" in str(exc_info.value)
    
    async def test_get_analysis_success(self, client, mock_api, mock_success_response):
        """Başarılı analiz sonucu alma testi."""
        analysis_id = "test-123"
        mock_api.get(
            f"{DEFAULT_BASE_URL}/analysis/{analysis_id}",
            payload=mock_success_response,
            status=200
        )
        
        result = await client.get_analysis(analysis_id)
        assert result["status"] == "success"
        assert result["analysis_id"] == analysis_id
    
    async def test_get_analysis_not_found(self, client, mock_api):
        """Bulunamayan analiz testi."""
        analysis_id = "nonexistent-id"
        mock_api.get(
            f"{DEFAULT_BASE_URL}/analysis/{analysis_id}",
            status=404
        )
        
        with pytest.raises(AnalysisError) as exc_info:
            await client.get_analysis(analysis_id)
        assert "Analiz bulunamadı" in str(exc_info.value)
    
    @pytest.mark.parametrize("invalid_id", ["", None, "   ", 123])
    async def test_get_analysis_invalid_id(self, client, invalid_id):
        """Geçersiz analiz ID testi."""
        with pytest.raises(ValidationError) as exc_info:
            await client.get_analysis(invalid_id)
        assert "Geçersiz analiz ID" in str(exc_info.value) 