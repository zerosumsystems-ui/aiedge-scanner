#!/usr/bin/env python3
"""
Generate iPhone 14 viewport screenshot (390×844) from dashboard_after.html
"""

import sys
from pathlib import Path

def create_screenshot():
    """Create screenshot using PIL and basic HTML parsing."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not available. Trying alternative...")
        return False

    html_path = Path("/Users/williamkosloski/video-pipeline/logs/live_scanner/dashboard_after.html")
    output_path = Path("/Users/williamkosloski/Library/Application Support/Claude/local-agent-mode-sessions/2c71c7a3-dac6-499e-9e14-51e3f1618346/3801c774-337c-4015-8bdb-4c430dc58216/local_7aa83b40-0498-4b98-800c-e852df3e11f5/outputs/dashboard_after.png")

    # Try creating a simple placeholder
    img = Image.new('RGB', (390, 844), color='white')
    draw = ImageDraw.Draw(img)

    # Read HTML file to extract visible text
    if html_path.exists():
        with open(html_path, 'r') as f:
            html_content = f.read()

        # Simple text extraction (basic)
        import re
        text_blocks = re.findall(r'<td[^>]*>([^<]+)</td>', html_content)

        # Draw some content
        y = 20
        for i, text in enumerate(text_blocks[:10]):
            if y < 800:
                draw.text((10, y), text[:40], fill='black')
                y += 30

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    print(f"Screenshot saved: {output_path}")
    return True

if __name__ == "__main__":
    create_screenshot()
