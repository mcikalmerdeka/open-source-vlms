"""
DeepSeek-OCR-2 Model Implementation
"""

import tempfile
import os
import sys
import shutil
import re
from io import StringIO
from typing import Dict, Any, List
from PIL import Image, ImageOps

from models.base import BaseOCRModel
from utils.image_utils import (
    extract_grounding_references, 
    draw_bounding_boxes, 
    clean_output, 
    embed_images
)

# Heavy imports deferred to load_model()
torch = None
spaces = None

class DeepSeekOCRModel(BaseOCRModel):
    """DeepSeek-OCR-2 model implementation"""
    
    TASK_PROMPTS = {
        "📋 Markdown": {"prompt": "<image>\n<|grounding|>Convert the document to markdown.", "has_grounding": True},
        "📝 Free OCR": {"prompt": "<image>\nFree OCR.", "has_grounding": False},
        "📍 Locate": {"prompt": "<image>\nLocate <|ref|>text<|/ref|> in the image.", "has_grounding": True},
        "🔍 Describe": {"prompt": "<image>\nDescribe this image in detail.", "has_grounding": False},
        "✏️ Custom": {"prompt": "", "has_grounding": False}
    }
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.base_size = 1024
        self.image_size = 768
        self.crop_mode = True
        
    def load_model(self):
        """Load DeepSeek-OCR-2 model"""
        global torch, spaces
        
        # Import heavy dependencies only when needed
        if torch is None:
            import torch as _torch
            torch = _torch
        if spaces is None:
            import spaces as _spaces
            spaces = _spaces
        
        from transformers import AutoModel, AutoTokenizer
        
        if not self.is_loaded:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['repo_id'], 
                trust_remote_code=True
            )
            # Try flash attention first, fall back to eager if not available
            try:
                self.model = AutoModel.from_pretrained(
                    self.config['repo_id'],
                    _attn_implementation='flash_attention_2',
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            except (ImportError, RuntimeError):
                # Flash attention not available, use eager attention
                self.model = AutoModel.from_pretrained(
                    self.config['repo_id'],
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            self.model = self.model.eval().cuda()
            self.is_loaded = True
    
    def get_task_options(self) -> List[str]:
        """Get available task options"""
        return list(self.TASK_PROMPTS.keys())
    
    def process_image(self, image: Image.Image, task: str, custom_prompt: str = "") -> Dict[str, Any]:
        """Process image with DeepSeek-OCR-2"""
        if image is None:
            return {"error": "No image provided"}
        
        if task in ["✏️ Custom", "📍 Locate"] and not custom_prompt.strip():
            return {"error": "Please enter a prompt"}
        
        # Preprocess image
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        image = ImageOps.exif_transpose(image)
        
        # Build prompt
        if task == "✏️ Custom":
            prompt = f"<image>\n{custom_prompt.strip()}"
            has_grounding = '<|grounding|>' in custom_prompt
        elif task == "📍 Locate":
            prompt = f"<image>\nLocate <|ref|>{custom_prompt.strip()}<|/ref|> in the image."
            has_grounding = True
        else:
            prompt = self.TASK_PROMPTS[task]["prompt"]
            has_grounding = self.TASK_PROMPTS[task]["has_grounding"]
        
        # Save temp image
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image.save(tmp.name, 'JPEG', quality=95)
        tmp.close()
        out_dir = tempfile.mkdtemp()
        
        # Capture stdout
        stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Run inference
        self.model.infer(
            tokenizer=self.tokenizer,
            prompt=prompt,
            image_file=tmp.name,
            output_path=out_dir,
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            save_results=False
        )
        
        # Get result
        debug_filters = ['PATCHES', '====', 'BASE:', 'directly resize', 'NO PATCHES', 'torch.Size', '%|']
        result = '\n'.join([l for l in sys.stdout.getvalue().split('\n')
                           if l.strip() and not any(s in l for s in debug_filters)]).strip()
        sys.stdout = stdout
        
        # Cleanup
        os.unlink(tmp.name)
        shutil.rmtree(out_dir, ignore_errors=True)
        
        if not result:
            return {"error": "No text detected"}
        
        # Process results
        cleaned = clean_output(result, False)
        markdown = clean_output(result, True)
        
        img_out = None
        crops = []
        
        if has_grounding and '<|ref|>' in result:
            refs = extract_grounding_references(result)
            if refs:
                img_out, crops = draw_bounding_boxes(image, refs, True)
        
        markdown = embed_images(markdown, crops)
        
        return {
            "text": cleaned,
            "markdown": markdown,
            "raw": result,
            "visualization": img_out,
            "crops": crops
        }
