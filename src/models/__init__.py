"""
Models module - Lazy imports to avoid loading heavy dependencies on import
"""

from models.base import BaseOCRModel

__all__ = [
    'BaseOCRModel',
    'get_model',
    'register_model'
]

# Registry with lazy loading
_MODEL_REGISTRY = {}

def _init_registry():
    """Initialize registry with lazy imports"""
    global _MODEL_REGISTRY
    if not _MODEL_REGISTRY:
        from models.deepseek_ocr import DeepSeekOCRModel
        from models.glm_ocr import GLMOCRModel
        from models.paddle_ocr import PaddleOCRModel
        
        _MODEL_REGISTRY = {
            'deepseek-ocr-2': DeepSeekOCRModel,
            'glm-ocr': GLMOCRModel,
            'paddleocr-vl-1.5': PaddleOCRModel
        }
    return _MODEL_REGISTRY

def get_model(model_id: str, config: dict) -> BaseOCRModel:
    """Factory function to get model instance - lazy loaded"""
    registry = _init_registry()
    model_class = registry.get(model_id)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_id}")
    return model_class(model_id, config)

def register_model(model_id: str, model_class):
    """Register a new model"""
    registry = _init_registry()
    registry[model_id] = model_class
