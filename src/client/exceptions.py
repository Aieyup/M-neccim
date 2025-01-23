class AgentClientError(Exception):
    """Temel Agent Client hatası."""
    pass

class APIConnectionError(AgentClientError):
    """API bağlantı hatası."""
    pass

class ValidationError(AgentClientError):
    """Veri doğrulama hatası."""
    pass

class AnalysisError(AgentClientError):
    """Analiz işlemi hatası."""
    pass 