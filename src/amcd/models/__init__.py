from . import cnn  # noqa: F401 — trigger registry registration
from .base import Model
from .cnn import CNNDenoisingModel

__all__ = ["Model", "CNNDenoisingModel"]
