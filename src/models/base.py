"""
Base model interface - all models must implement this interface
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image

class BaseOCRModel(ABC):
    """Base class for all OCR models"""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self):
        """Load the model and necessary components"""
        pass
    
    @abstractmethod
    def process_image(self, image: Image.Image, task: str, **kwargs) -> Dict[str, Any]:
        """
        Process a single image
        
        Returns:
            Dict with keys like:
            - 'text': extracted text
            - 'markdown': markdown formatted output
            - 'visualization': image with bounding boxes (if applicable)
            - 'crops': list of cropped images (if applicable)
            - 'raw': raw model output
        """
        pass
    
    @abstractmethod
    def get_task_options(self) -> List[str]:
        """Get available task options for this model"""
        pass
    
    def supports_pdf(self) -> bool:
        """Check if model supports PDF processing"""
        return self.config.get('supports_pdf', False)
    
    def supports_grounding(self) -> bool:
        """Check if model supports bounding box visualization"""
        return self.config.get('supports_grounding', False)
