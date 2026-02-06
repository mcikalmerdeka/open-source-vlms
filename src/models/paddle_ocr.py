"""
PaddleOCR-VL-1.5 Model Implementation
"""

import os
import base64
import json
import re
from typing import Dict, Any, List, Tuple
from PIL import Image
import requests
from urllib.parse import urlparse
import gradio as gr

from models.base import BaseOCRModel

class PaddleOCRModel(BaseOCRModel):
    """PaddleOCR-VL-1.5 model implementation using API"""
    
    TASK_OPTIONS = {
        "📄 Document Parsing": "document_parsing",
        "🎯 Element Recognition": "element_recognition",
        "🔍 Spotting": "spotting"
    }
    
    ELEMENT_OPTIONS = {
        "📝 Text Recognition": "ocr",
        "🔢 Formula Recognition": "formula",
        "📊 Table Recognition": "table",
        "📈 Chart Recognition": "chart",
        "🔍 Spotting": "spotting",
        "🔏 Seal Recognition": "seal"
    }
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.api_url = os.environ.get("PADDLEOCR_API_URL")
        self.token = os.environ.get("PADDLEOCR_TOKEN")
        
    def load_model(self):
        """PaddleOCR uses API, no model loading needed"""
        self.is_loaded = True
    
    def get_task_options(self) -> List[str]:
        """Get available task options"""
        return list(self.TASK_OPTIONS.keys())
    
    def get_element_options(self) -> List[str]:
        """Get element recognition options"""
        return list(self.ELEMENT_OPTIONS.keys())
    
    def _file_to_b64(self, path_or_url: str) -> Tuple[str, int]:
        """Convert file to base64"""
        is_url = isinstance(path_or_url, str) and path_or_url.startswith(("http://", "https://"))
        
        if is_url:
            r = requests.get(path_or_url, timeout=600)
            r.raise_for_status()
            content = r.content
            ext = os.path.splitext(urlparse(path_or_url).path)[1].lower()
        else:
            ext = os.path.splitext(path_or_url)[1].lower()
            with open(path_or_url, "rb") as f:
                content = f.read()
        
        return base64.b64encode(content).decode("utf-8"), 1
    
    def _call_api(self, path_or_url: str, task_type: str, element_type: str = None,
                  use_chart: bool = False, use_unwarp: bool = True, 
                  use_orient: bool = True) -> Dict[str, Any]:
        """Call PaddleOCR API"""
        
        if not self.api_url:
            return {"error": "PaddleOCR API URL not configured"}
        
        is_url = isinstance(path_or_url, str) and path_or_url.startswith(("http://", "https://"))
        
        if is_url:
            payload = {
                "file": path_or_url,
                "useLayoutDetection": task_type == "document_parsing",
                "useDocUnwarping": use_unwarp,
                "useDocOrientationClassify": use_orient
            }
        else:
            b64, file_type = self._file_to_b64(path_or_url)
            payload = {
                "file": b64,
                "useLayoutDetection": task_type == "document_parsing",
                "fileType": file_type,
                "useDocUnwarping": use_unwarp,
                "useDocOrientationClassify": use_orient
            }
        
        if task_type != "document_parsing" and element_type:
            payload["promptLabel"] = element_type
            payload["useDocUnwarping"] = False
            payload["useDocOrientationClassify"] = False
        
        if task_type == "document_parsing" and use_chart:
            payload["useChartRecognition"] = True
        
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"bearer {self.token}"
        
        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=600)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}
        
        if data.get("errorCode", -1) != 0:
            return {"error": "API returned an error"}
        
        return data
    
    def _process_response(self, data: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Process API response"""
        result = data.get("result", {})
        layout_results = result.get("layoutParsingResults", [])
        
        if not layout_results:
            return {
                "text": "No content was recognized.",
                "markdown": "No content was recognized.",
                "raw": json.dumps(result, ensure_ascii=False, indent=2),
                "visualization": None,
                "crops": []
            }
        
        page0 = layout_results[0] or {}
        md_data = page0.get("markdown") or {}
        md_text = md_data.get("text", "") or ""
        md_images_map = md_data.get("images", {})
        
        # Replace image placeholders with actual URLs
        if md_images_map:
            for placeholder_path, image_url in md_images_map.items():
                md_text = md_text.replace(f'src="{placeholder_path}"', f'src="{image_url}"') \
                                 .replace(f']({placeholder_path})', f']({image_url})')
        
        # Get visualization image
        out_imgs = page0.get("outputImages") or {}
        vis_image = None
        
        if task_type == "spotting":
            vis_image = out_imgs.get("spotting_res_img")
        elif len(out_imgs) >= 2:
            sorted_urls = [img_url for _, img_url in sorted(out_imgs.items()) if img_url]
            vis_image = sorted_urls[1] if len(sorted_urls) >= 2 else sorted_urls[0] if sorted_urls else None
        else:
            sorted_urls = [img_url for _, img_url in sorted(out_imgs.items()) if img_url]
            vis_image = sorted_urls[0] if sorted_urls else None
        
        # Escape inequalities in math
        md_text = self._escape_inequalities_in_math(md_text)
        
        # For spotting, get JSON result
        if task_type == "spotting":
            pruned = page0.get("prunedResult") or {}
            spotting_res = pruned.get("spotting_res") or {}
            return {
                "text": json.dumps(spotting_res, ensure_ascii=False, indent=2),
                "markdown": md_text,
                "raw": json.dumps(result, ensure_ascii=False, indent=2),
                "visualization": vis_image,
                "crops": [],
                "json_result": spotting_res
            }
        
        return {
            "text": md_text,
            "markdown": md_text,
            "raw": json.dumps(result, ensure_ascii=False, indent=2),
            "visualization": vis_image,
            "crops": []
        }
    
    def _escape_inequalities_in_math(self, md: str) -> str:
        """Escape inequalities in math expressions"""
        _MATH_PATTERNS = [
            re.compile(r"\$\$([\s\S]+?)\$\$"),
            re.compile(r"\$([^\$]+?)\$"),
            re.compile(r"\\\[([\s\S]+?)\\\]"),
            re.compile(r"\\\(([\s\S]+?)\\\)"),
        ]
        
        def fix(s: str) -> str:
            s = s.replace("<=", r" \le ").replace(">=", r" \ge ")
            s = s.replace("≤", r" \le ").replace("≥", r" \ge ")
            s = s.replace("<", r" \lt ").replace(">", r" \gt ")
            return s
        
        for pat in _MATH_PATTERNS:
            md = pat.sub(lambda m: m.group(0).replace(m.group(1), fix(m.group(1))), md)
        return md
    
    def process_image(self, image_path: str, task: str, element_type: str = None,
                     use_chart: bool = False, use_unwarp: bool = True, 
                     use_orient: bool = True, **kwargs) -> Dict[str, Any]:
        """Process image with PaddleOCR"""
        
        if not image_path:
            return {"error": "No image provided"}
        
        task_type = self.TASK_OPTIONS.get(task, "document_parsing")
        element_label = self.ELEMENT_OPTIONS.get(element_type) if element_type else None
        
        data = self._call_api(
            image_path, 
            task_type, 
            element_label,
            use_chart, 
            use_unwarp, 
            use_orient
        )
        
        if "error" in data:
            return data
        
        return self._process_response(data, task_type)
