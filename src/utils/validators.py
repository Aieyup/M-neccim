import numpy as np
from typing import Any, Union
from ..client.exceptions import ValidationError

def validate_numpy_array(data: Any) -> np.ndarray:
    """Numpy array validasyonu yapar."""
    if not isinstance(data, np.ndarray):
        raise ValidationError("Veri numpy array formatında olmalıdır")
    
    if data.size == 0:
        raise ValidationError("Veri boş olamaz")
        
    if not np.isfinite(data).all():
        raise ValidationError("Veri içinde sonsuz veya NaN değerler var")
    
    return data

def validate_analysis_id(analysis_id: Union[str, None]) -> str:
    """Analiz ID validasyonu yapar."""
    if not analysis_id or not isinstance(analysis_id, str):
        raise ValidationError("Geçersiz analiz ID")
    return analysis_id.strip() 