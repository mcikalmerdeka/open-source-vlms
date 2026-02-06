"""
Unified OCR Platform
A Gradio-based platform that supports multiple Vision-Language/OCR models:
- DeepSeek-OCR-2
- GLM-OCR  
- PaddleOCR-VL-1.5

This platform is designed to be easily extensible - you can add or remove models
by updating the src/models/ directory and config.py.
"""

import gradio as gr
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import AVAILABLE_MODELS, MODEL_CONFIGS
from interface import create_interface

def main():
    """Main entry point for the application"""
    demo = create_interface()
    
    # Launch the app
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
