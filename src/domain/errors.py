class DomainError(Exception):
    """Base class for domain exceptions."""
    pass

class DataFetchError(DomainError):
    """Raised when there is an issue fetching market data."""
    def __init__(self, reason: str) -> None:
        super().__init__(f"Data fetch error: {reason}")

class SimulationError(DomainError):
    """Raised when simulation fails."""
    def __init__(self, reason: str) -> None:
        super().__init__(f"Simulation error: {reason}")
