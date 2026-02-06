"""
Utility functions for image processing and visualization
"""

import re
import base64
from io import BytesIO
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def extract_grounding_references(text: str) -> List[Tuple[str, str, str]]:
    """Extract grounding references from model output"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    return re.findall(pattern, text, re.DOTALL)

def draw_bounding_boxes(
    image: Image.Image, 
    refs: List[Tuple[str, str, str]], 
    extract_images: bool = False
) -> Tuple[Image.Image, List[Image.Image]]:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input PIL Image
        refs: List of (full_match, label, coords) tuples
        extract_images: Whether to extract cropped images
        
    Returns:
        Tuple of (annotated_image, list_of_crops)
    """
    img_w, img_h = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    # Try to load font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()
    
    crops = []
    color_map = {}
    np.random.seed(42)
    
    for ref in refs:
        label = ref[1]
        if label not in color_map:
            color_map[label] = (
                np.random.randint(50, 255), 
                np.random.randint(50, 255), 
                np.random.randint(50, 255)
            )
        
        color = color_map[label]
        
        # Parse coordinates (format: [[x1, y1, x2, y2], ...])
        try:
            coords = eval(ref[2])
        except:
            continue
            
        color_a = color + (60,)
        
        for box in coords:
            try:
                x1, y1, x2, y2 = int(box[0]/999*img_w), int(box[1]/999*img_h), int(box[2]/999*img_w), int(box[3]/999*img_h)
                
                if extract_images and label == 'image':
                    crops.append(image.crop((x1, y1, x2, y2)))
                
                width = 5 if label == 'title' else 3
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                draw2.rectangle([x1, y1, x2, y2], fill=color_a)
                
                # Draw label
                text_bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                ty = max(0, y1 - 20)
                draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
                draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))
            except:
                continue
    
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, crops

def clean_output(text: str, include_images: bool = False) -> str:
    """Clean model output by removing special tokens"""
    if not text:
        return ""
    
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    img_num = 0
    
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            if include_images:
                text = text.replace(match[0], f'\n\n**[Figure {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(match[0], '', 1)
        else:
            text = re.sub(rf'(?m)^[^\n]*{re.escape(match[0])}[^\n]*\n?', '', text)
    
    # Clean up math symbols
    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    
    return text.strip()

def embed_images(markdown: str, crops: List[Image.Image]) -> str:
    """Embed cropped images as base64 in markdown"""
    if not crops:
        return markdown
    
    for i, img in enumerate(crops):
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        markdown = markdown.replace(
            f'**[Figure {i + 1}]**', 
            f'\n\n![Figure {i + 1}](data:image/png;base64,{b64})\n\n', 
            1
        )
    return markdown

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
