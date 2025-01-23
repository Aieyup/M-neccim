import numpy as np
from typing import Tuple, List, Dict, Any
import json

def prepare_clusters(n_samples: int, n_clusters: int = 2, seed: int = 42) -> np.ndarray:
    """Test veya örnek veri kümesi oluşturur."""
    np.random.seed(seed)
    clusters = []
    
    for i in range(n_clusters):
        cluster = np.random.normal(i*3, 1, (n_samples // n_clusters, 2))
        clusters.append(cluster)
    
    return np.vstack(clusters)

def format_analysis_result(result: Dict[str, Any]) -> str:
    """Analiz sonuçlarını formatlar."""
    return json.dumps(result, indent=2, ensure_ascii=False) 