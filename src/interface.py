"""
Main Gradio Interface
"""

import gradio as gr
import os
import sys
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

# Import models and config
from config import AVAILABLE_MODELS, MODEL_CONFIGS, MODEL_DISPLAY_NAMES
from models import get_model

# Global model cache
_model_cache = {}
_current_model_id = None

def get_or_load_model(model_id: str):
    """Get or load model with caching"""
    global _current_model_id
    
    if model_id not in _model_cache:
        config = MODEL_CONFIGS[model_id]
        model = get_model(model_id, config)
        try:
            model.load_model()
        except RuntimeError as e:
            if "GPU" in str(e) or "CUDA" in str(e):
                # GPU not available, mark as unavailable
                model.is_loaded = False
                print(f"⚠️ {model_id} requires GPU: {e}")
            else:
                raise
        _model_cache[model_id] = model
    
    _current_model_id = model_id
    return _model_cache[model_id]

def load_image(file_path, page_num=1):
    """Load image from file or PDF page"""
    if not file_path:
        return None
    
    if file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        page_idx = max(0, min(int(page_num) - 1, len(doc) - 1))
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        doc.close()
        return img
    else:
        return Image.open(file_path)

def get_pdf_page_count(file_path):
    """Get PDF page count"""
    if not file_path or not file_path.lower().endswith('.pdf'):
        return 1
    doc = fitz.open(file_path)
    count = len(doc)
    doc.close()
    return count

def update_model_selection(model_id):
    """Update UI when model is selected"""
    config = MODEL_CONFIGS[model_id]
    
    # Build description
    desc = f"## {MODEL_DISPLAY_NAMES[model_id]}\n\n{config['description']}"
    
    # Show/hide PDF selector based on model support
    pdf_visible = config.get('supports_pdf', False)
    
    # Get task options from config (no need to load model)
    tasks = list(config.get('tasks', {}).values())
    
    # Reset task to first option of new model
    return [
        gr.Dropdown(choices=tasks, value=tasks[0] if tasks else None, interactive=True),
        gr.Number(visible=pdf_visible),
        gr.File(visible=pdf_visible),
        desc
    ]

def toggle_custom_prompt(model_id, task):
    """Toggle custom prompt visibility"""
    if model_id == "deepseek-ocr-2":
        if task in ["✏️ Custom", "📍 Locate"]:
            return gr.update(visible=True, label="Custom Prompt" if task == "✏️ Custom" else "Text to Locate")
    return gr.update(visible=False)

def process_with_model(model_id, image, file_path, task, custom_prompt, page_num,
                       element_type, use_chart, use_unwarp, use_orient):
    """Process image with selected model"""
    
    # Get source image
    if file_path:
        src = file_path
        img = load_image(file_path, page_num)
    elif image is not None:
        src = None
        img = image
    else:
        return ["Error: Please upload an image or file", "", "", None, []]
    
    # Get model
    model = get_or_load_model(model_id)
    
    # Process based on model type
    if model_id == "deepseek-ocr-2":
        result = model.process_image(img, task, custom_prompt)
    elif model_id == "glm-ocr":
        result = model.process_image(img, task)
    elif model_id == "paddleocr-vl-1.5":
        # For PaddleOCR, we need the file path
        src_path = src if src else None
        if src_path is None and img is not None:
            # Save temp image
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            img.save(tmp.name, 'PNG')
            tmp.close()
            src_path = tmp.name
        
        result = model.process_image(
            src_path, task, element_type,
            use_chart, use_unwarp, use_orient
        )
        
        # Cleanup temp file
        if src_path and src_path.startswith(tempfile.gettempdir()):
            os.unlink(src_path)
    else:
        return ["Error: Unknown model", "", "", None, []]
    
    # Handle errors
    if "error" in result:
        return [result["error"], "", "", None, []]
    
    # Extract results
    text = result.get("text", "")
    markdown = result.get("markdown", "")
    raw = result.get("raw", "")
    visualization = result.get("visualization")
    crops = result.get("crops", [])
    
    # Convert visualization to PIL if it's a URL (for PaddleOCR)
    if isinstance(visualization, str) and visualization.startswith("http"):
        import requests
        try:
            r = requests.get(visualization, timeout=30)
            if r.status_code == 200:
                visualization = Image.open(BytesIO(r.content))
        except:
            visualization = None
    
    return [text, markdown, raw, visualization, crops]

def create_interface():
    """Create the Gradio interface"""
    
    # Get examples
    examples = []
    examples_dir = "examples"
    if os.path.exists(examples_dir):
        for f in os.listdir(examples_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                examples.append(os.path.join(examples_dir, f))
    
    with gr.Blocks(title="Unified OCR Platform") as demo:
        gr.Markdown("""
        # 🔍 VLMs OCR Playground
        
        **Compare and use multiple state-of-the-art Vision-Language OCR models:**
        - **DeepSeek-OCR-2**: Document to markdown with layout detection  
        - **GLM-OCR**: Specialized recognition for text, formulas, and tables  
        - **PaddleOCR-VL-1.5**: Full-page document parsing with layout detection
        
        Select a model below to get started!
        """)
        
        # Status notice
        gr.Markdown("""
        > ⚠️ **Current Deployment Status:**
        > - 🚀 **DeepSeek-OCR-2**: Requires GPU - temporarily unavailable on this CPU-only deployment
        > - 🔮 **GLM-OCR**: Available ✅
        > - 📄 **PaddleOCR-VL-1.5**: Available ✅ (requires API key)
        """)
        
        # Model selection - default to PaddleOCR-VL-1.5 (index 2)
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=[(MODEL_DISPLAY_NAMES[m], m) for m in AVAILABLE_MODELS],
                value=AVAILABLE_MODELS[2],  # Default to PaddleOCR-VL-1.5
                label="Select OCR Model",
                scale=1
            )
        
        # Model description
        model_desc = gr.Markdown(MODEL_CONFIGS[AVAILABLE_MODELS[2]]['description'])
        
        with gr.Row():
            # Left column - inputs
            with gr.Column(scale=1):
                # File upload
                file_input = gr.File(
                    label="Upload Image or PDF",
                    file_types=["image", ".pdf"],
                    type="filepath"
                )
                
                # Image preview
                input_image = gr.Image(label="Input", type="pil", height=300)
                
                # PDF page selector (only for DeepSeek)
                page_selector = gr.Number(
                    label="Select Page",
                    value=1,
                    minimum=1,
                    step=1,
                    visible=False
                )
                
                # Task selector - initialize with PaddleOCR's tasks (index 2, same as model_selector default)
                initial_tasks = list(MODEL_CONFIGS[AVAILABLE_MODELS[2]]['tasks'].values())
                task_selector = gr.Dropdown(
                    label="Task",
                    choices=initial_tasks,
                    value=initial_tasks[0] if initial_tasks else None
                )
                
                # Custom prompt (for DeepSeek)
                custom_prompt = gr.Textbox(
                    label="Custom Prompt",
                    lines=2,
                    visible=False
                )
                
                # Element type selector (for PaddleOCR)
                element_selector = gr.Dropdown(
                    label="Element Type",
                    choices=["📝 Text Recognition", "🔢 Formula Recognition", 
                            "📊 Table Recognition", "📈 Chart Recognition",
                            "🔍 Spotting", "🔏 Seal Recognition"],
                    value="📝 Text Recognition",
                    visible=False
                )
                
                # PaddleOCR options
                with gr.Row(visible=False) as paddle_options:
                    chart_checkbox = gr.Checkbox(label="Chart Parsing", value=False)
                    unwarp_checkbox = gr.Checkbox(label="Doc Unwarping", value=True)
                    orient_checkbox = gr.Checkbox(label="Orientation Fix", value=True)
                
                # Process button
                process_btn = gr.Button("🔍 Process", variant="primary", size="lg")
                
                # Examples
                if examples:
                    gr.Examples(
                        examples=[[ex] for ex in examples[:5]],
                        inputs=[input_image],
                        label="Example Images"
                    )
            
            # Right column - outputs
            with gr.Column(scale=2):
                with gr.Tabs() as output_tabs:
                    with gr.Tab("📝 Text"):
                        text_output = gr.Textbox(
                            label="Extracted Text",
                            lines=20
                        )
                    
                    with gr.Tab("📄 Markdown"):
                        markdown_output = gr.Markdown()
                    
                    with gr.Tab("🖼️ Visualization"):
                        vis_output = gr.Image(label="Bounding Boxes", height=500)
                    
                    with gr.Tab("🎨 Crops"):
                        gallery_output = gr.Gallery(
                            label="Extracted Images",
                            columns=3,
                            height=400
                        )
                    
                    with gr.Tab("🔧 Raw"):
                        raw_output = gr.Textbox(
                            label="Raw Output",
                            lines=20
                        )
        
        # Info section
        with gr.Accordion("ℹ️ About Models", open=False):
            gr.Markdown("""
            ### Model Details
            
            **DeepSeek-OCR-2**
            - Converts documents to structured markdown
            - Supports layout detection with bounding boxes
            - Can locate specific text in images
            - Supports PDF processing
            
            **GLM-OCR**
            - Specialized for text, formula, and table recognition
            - Clean output for document understanding
            - Optimized for academic and scientific documents
            
            **PaddleOCR-VL-1.5**
            - Full-page document parsing with layout detection
            - Element-level recognition (text, tables, charts, formulas)
            - Spotting capability for locating elements
            - Requires API configuration
            """)
        
        # Event handlers
        model_selector.change(
            update_model_selection,
            [model_selector],
            [task_selector, page_selector, file_input, model_desc]
        )
        
        model_selector.change(
            lambda m: gr.update(visible=m == "paddleocr-vl-1.5"),
            [model_selector],
            [element_selector]
        )
        
        model_selector.change(
            lambda m: gr.update(visible=m == "paddleocr-vl-1.5"),
            [model_selector],
            [paddle_options]
        )
        
        task_selector.change(
            toggle_custom_prompt,
            [model_selector, task_selector],
            [custom_prompt]
        )
        
        task_selector.change(
            lambda m, t: gr.update(visible=m == "paddleocr-vl-1.5" and t == "🎯 Element Recognition"),
            [model_selector, task_selector],
            [element_selector]
        )
        
        file_input.change(
            lambda fp: load_image(fp, 1) if fp else None,
            [file_input],
            [input_image]
        )
        
        file_input.change(
            lambda fp: gr.update(
                visible=fp.lower().endswith('.pdf') if fp else False,
                maximum=get_pdf_page_count(fp) if fp and fp.lower().endswith('.pdf') else 1,
                value=1
            ),
            [file_input],
            [page_selector]
        )
        
        page_selector.change(
            lambda fp, pn: load_image(fp, pn) if fp else None,
            [file_input, page_selector],
            [input_image]
        )
        
        process_btn.click(
            process_with_model,
            [model_selector, input_image, file_input, task_selector, custom_prompt, page_selector,
             element_selector, chart_checkbox, unwarp_checkbox, orient_checkbox],
            [text_output, markdown_output, raw_output, vis_output, gallery_output]
        )
        
        # Initialize with first model
        demo.load(
            update_model_selection,
            [model_selector],
            [task_selector, page_selector, file_input, model_desc]
        )
    
    return demo
