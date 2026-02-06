"""
GLM-OCR Model Implementation
"""

import tempfile
import os
from typing import Dict, Any, List
from PIL import Image, ImageOps

from models.base import BaseOCRModel

# Heavy imports deferred to load_model()
torch = None
spaces = None

class GLMOCRModel(BaseOCRModel):
    """GLM-OCR model implementation"""
    
    TASK_PROMPTS = {
        "📝 Text Recognition": "Text Recognition:",
        "🔢 Formula Recognition": "Formula Recognition:",
        "📊 Table Recognition": "Table Recognition:"
    }
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        
    def load_model(self):
        """Load GLM-OCR model"""
        global torch, spaces
        
        # Import heavy dependencies only when needed
        if torch is None:
            import torch as _torch
            torch = _torch
        if spaces is None:
            import spaces as _spaces
            spaces = _spaces
        
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        if not self.is_loaded:
            self.processor = AutoProcessor.from_pretrained(
                self.config['repo_id'],
                trust_remote_code=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=self.config['repo_id'],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.is_loaded = True
    
    def get_task_options(self) -> List[str]:
        """Get available task options"""
        return list(self.TASK_PROMPTS.keys())
    
    def process_image(self, image: Image.Image, task: str, **kwargs) -> Dict[str, Any]:
        """Process image with GLM-OCR"""
        if image is None:
            return {"error": "No image provided"}
        
        # Preprocess image
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        image = ImageOps.exif_transpose(image)
        
        # Save temp image
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        image.save(tmp.name, 'PNG')
        tmp.close()
        
        # Get prompt
        prompt = self.TASK_PROMPTS.get(task, "Text Recognition:")
        
        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": tmp.name},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        inputs.pop("token_type_ids", None)
        
        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Cleanup
        os.unlink(tmp.name)
        
        result = output_text.strip()
        
        return {
            "text": result,
            "markdown": result,
            "raw": result,
            "visualization": None,
            "crops": []
        }
