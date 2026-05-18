"""v0.1 compat shim — LightningStore is the real store now."""
from .store import LightningStore as RunStore
__all__ = ["RunStore"]
