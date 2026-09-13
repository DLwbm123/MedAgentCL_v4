"""Med-PRISM v2: task-boundary repair, never a gradient-training objective."""
from .analytic_transport import TransportResult, transport_keys
from .config import TPMConfig

__all__ = ["TPMConfig", "TransportResult", "transport_keys"]
