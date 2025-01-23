import logging
from pathlib import Path

def setup_logger():
    """Logger yapılandırmasını oluşturur."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logger() 