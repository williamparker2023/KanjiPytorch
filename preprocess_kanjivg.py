import os
import json
import cairosvg
from PIL import Image

# Directory where the Kanji SVG files are located
SVG_DIR = 'kanji'

# Directory to save the preprocessed images
OUTPUT_DIR = 'processed_kanji_images'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the Kanji index JSON
with open('kvg-index.json', 'r', encoding='utf-8') as f:
    kanji_index = json.load(f)

def svg_to_png(svg_file_path, output_file_path):
    """
    Convert an SVG file to PNG using cairosvg.
    """
    cairosvg.svg2png(url=svg_file_path, write_to=output_file_path)

    # Open the generated PNG and perform any additional processing, such as resizing
    image = Image.open(output_file_path)
    image = image.convert('RGB')
    image.save(output_file_path, 'PNG')

def preprocess_kanji():
    """
    Preprocess Kanji SVG files, convert them to PNG, resize, and save with labels.
    """
    for kanji, svg_files in kanji_index.items():
        for svg_file in svg_files:
            svg_path = os.path.join(SVG_DIR, svg_file)
            png_file = f'{kanji}_{svg_file.split(".")[0]}.png'
            png_path = os.path.join(OUTPUT_DIR, png_file)

            # Convert SVG to PNG
            svg_to_png(svg_path, png_path)

            # Resize the PNG to a fixed size (e.g., 64x64)
            image = Image.open(png_path)
            image = image.resize((64, 64))
            image.save(png_path)

            print(f"Processed {kanji} ({svg_file}) and saved to {png_path}")

if __name__ == "__main__":
    preprocess_kanji()
