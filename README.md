# Unified OCR Platform

A unified Gradio-based platform for comparing and using multiple state-of-the-art Vision-Language OCR models. Deploy easily to Hugging Face Spaces.

## Supported Models

1. **DeepSeek-OCR-2** - Document to markdown with layout detection and PDF support
2. **GLM-OCR** - Specialized recognition for text, formulas, and tables
3. **PaddleOCR-VL-1.5** - Full-page document parsing with layout detection

## Project Structure

```
.
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── src/                    # Source code
│   ├── config.py          # Model configurations
│   ├── interface.py       # Gradio interface
│   ├── models/            # Model implementations
│   └── utils/             # Utilities
└── examples/              # Sample images
```

## Setup

### Option 1: Using pip

```bash
pip install -r requirements.txt
python main.py
```

### Option 2: Using uv (Recommended for faster installs)

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
python main.py
```

### Option 3: Using conda

```bash
conda create -n ocr python=3.10
conda activate ocr
pip install -r requirements.txt
python main.py
```

## Testing Without Heavy Dependencies

You can test the UI structure without downloading models:

```bash
# Test imports only
python test_imports.py

# Test UI creation (requires gradio only)
pip install gradio
python -c "from interface import create_interface; demo = create_interface()"
```

## Usage

1. Select a model from the dropdown
2. Upload an image or PDF (DeepSeek only)
3. Choose a task
4. Click Process
5. View results in tabs (Text, Markdown, Visualization, Crops, Raw)

## Adding New Models

To add a new model:

1. Create model class in `src/models/`
2. Register in `src/models/__init__.py`
3. Add config in `src/config.py`

See `src/models/base.py` for the interface.

## Troubleshooting

### Dependency Conflicts

If you see errors about `tokenizers` or `transformers` version conflicts:
- The requirements.txt uses flexible version ranges to avoid conflicts
- Remove `uv.lock` or `poetry.lock` if present and reinstall
- Use `pip install -r requirements.txt --force-reinstall` if needed

### Flash Attention Not Available

The models will automatically fall back to standard attention if flash-attn is not installed. This is normal and won't affect functionality.

### Out of Memory

If you run out of GPU memory:
- Run on CPU (slower but works): The models will automatically use CPU if GPU is not available
- Process smaller images
- Use one model at a time

## Environment Variables

- `PADDLEOCR_API_URL`: API endpoint for PaddleOCR (optional)
- `PADDLEOCR_TOKEN`: Authentication token for PaddleOCR (optional)

## License

This project combines multiple open-source OCR models. Please refer to individual model licenses.
