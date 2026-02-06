"""
Configuration for the Unified OCR Platform
"""

# Available models - easy to add/remove models here
AVAILABLE_MODELS = [
    "deepseek-ocr-2",
    "glm-ocr",
    "paddleocr-vl-1.5"
]

# Model display names
MODEL_DISPLAY_NAMES = {
    "deepseek-ocr-2": "🚀 DeepSeek-OCR-2",
    "glm-ocr": "🔮 GLM-OCR",
    "paddleocr-vl-1.5": "📄 PaddleOCR-VL-1.5"
}

# Model configurations
MODEL_CONFIGS = {
    "deepseek-ocr-2": {
        "name": "DeepSeek-OCR-2",
        "repo_id": "deepseek-ai/DeepSeek-OCR-2",
        "description": "Convert documents to markdown, extract text, parse figures, and locate specific content with bounding boxes.",
        "supports_pdf": True,
        "supports_grounding": True,
        "tasks": {
            "markdown": "📋 Markdown",
            "free_ocr": "📝 Free OCR",
            "locate": "📍 Locate",
            "describe": "🔍 Describe",
            "custom": "✏️ Custom"
        }
    },
    "glm-ocr": {
        "name": "GLM-OCR",
        "repo_id": "zai-org/GLM-OCR",
        "description": "A multimodal OCR model for complex document understanding with specialized recognition modes.",
        "supports_pdf": False,
        "supports_grounding": False,
        "tasks": {
            "text": "📝 Text Recognition",
            "formula": "🔢 Formula Recognition",
            "table": "📊 Table Recognition"
        }
    },
    "paddleocr-vl-1.5": {
        "name": "PaddleOCR-VL-1.5",
        "repo_id": "PaddlePaddle/PaddleOCR-VL-1.5",
        "description": "Full-page document parsing with layout detection and element-level recognition.",
        "supports_pdf": False,
        "supports_grounding": True,
        "tasks": {
            "document_parsing": "📄 Document Parsing",
            "element_recognition": "🎯 Element Recognition",
            "spotting": "🔍 Spotting"
        }
    }
}

# Global settings
BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True
