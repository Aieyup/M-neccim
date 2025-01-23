import pytest
import numpy as np
from pathlib import Path
import os
from typing import Generator
from aioresponses import aioresponses

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Test veri dizinini döndürür."""
    return Path(__file__).parent / "test_data"

@pytest.fixture(autouse=True)
def setup_test_env(test_data_dir: Path) -> None:
    """Test ortamını hazırlar."""
    os.environ["TESTING"] = "1"
    test_data_dir.mkdir(exist_ok=True)
    yield
    os.environ.pop("TESTING", None)

@pytest.fixture
def mock_api() -> Generator:
    """API çağrılarını mocklamak için fixture."""
    with aioresponses() as m:
        yield m 