import os
import numpy as np
import shutil
import sys
from spectral import envi
from pycocotools.coco import COCO
from PIL import Image

# ====== 💡 Set only this ======
#CLASS_NAME = 'C1'
IMG_DIR = '/home/pablosetra/Pictures/imagenes-higos/raw/julio'
ANNOTATIONS_BASE = './outputs/segm-masks-2023-july-17-28'
OUTPUT_BASE = './outputs/cropped-hypercubes'
# ==============================


def radiometric_correction(image, white, black):
    """Apply radiometric correction: (image - black) / (white - black)"""
    
    # Convert to float32 for precision
    white_f = white.astype(np.float32)
    black_f = black.astype(np.float32)
    image_f = image.astype(np.float32)

    # Avoid division by zero
    eps = 1e-6
    diff = white_f - black_f
    diff[diff < eps] = eps

    # Perform correction
    corrected = (image_f - black_f) / diff

    # Make sure values are in range [0, 1] and scale to [0, 255]
    corrected = np.clip(corrected, 0, 1) * 255.0

    return corrected.astype(np.uint8)


def find_image_recursive(img_dir, file_name):
    for root, _, files in os.walk(img_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    raise FileNotFoundError(f"{file_name} not found in {img_dir}")


def find_hypercube_files_from_path(image_path):
    """
    Given a full image path, returns paths to the associated HDR files in its sibling 'capture' folder.
    """
    file_name = os.path.basename(image_path)
    basename = file_name.replace('.png', '')
    parent_dir = os.path.dirname(image_path)
    capture_dir = os.path.join(parent_dir, 'capture')

    if not os.path.isdir(capture_dir):
        print(f"❌ 'capture' folder not found in {parent_dir}")
        return None

    paths = {
        'image': os.path.join(capture_dir, f'{basename}.hdr'),
        'white': os.path.join(capture_dir, f'WHITEREF_{basename}.hdr'),
        'dark': os.path.join(capture_dir, f'DARKREF_{basename}.hdr'),
    }

    if all(os.path.exists(p) for p in paths.values()):
        return paths
    else:
        print(f"❌ One or more HDR files missing in {capture_dir}")
        return None


def load_corrected_hypercube(paths: dict) -> np.ndarray:
    """
    Loads hyperspectral cube and applies row-wise radiometric correction using averaged white/black lines.
    Handles shape mismatches safely.
    Returns corrected cube as uint8.
    """

    # Load hyperspectral data
    img = envi.open(paths['image']).load()
    white = envi.open(paths['white']).load()
    dark = envi.open(paths['dark']).load()

    # Average white and dark references over rows
    white_avg = np.mean(white, axis=0, keepdims=True)  # shape: (1, cols, bands)
    dark_avg = np.mean(dark, axis=0, keepdims=True)

    # Apply correction line by line
    corrected_data = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        corrected_data[i, :, :] = radiometric_correction(img[i, :, :], white_avg, dark_avg)

    return corrected_data

def save_rgb_composite(cube, output_path='rgb_composite.png', r=60, g=30, b=10):
    """
    Saves an RGB composite image using 3 selected bands from a hyperspectral cube.

    Parameters:
    - cube: NumPy array of shape (rows, cols, bands), dtype uint8
    - output_path: path to save the resulting image
    - r, g, b: indices of bands to use for R, G, B channels
    """
    # Ensure the cube has enough bands
    assert cube.ndim == 3 and cube.shape[2] > max(r, g, b), "Cube has insufficient bands"

    rgb = cube[:, :, [r, g, b]]

    # Normalize to 0–255 if not already uint8
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    # Save image
    Image.fromarray(rgb).save(output_path)
    print(f"✅ RGB composite saved to: {output_path}")


def extract_cropped_cubes(cube, coco, img_info, anns, output_dir='cropped_cubes'):
    """
    Extrae y guarda subcubos hiperespectrales corregidos usando bbox + máscaras de COCO.
    Guarda todos los archivos en la misma carpeta con nombres: <image_name>_ann<id>.npy
    """
    os.makedirs(output_dir, exist_ok=True)
    file_base = os.path.splitext(img_info['file_name'])[0]

    for ann in anns:
        ann_id = ann['id']
        x, y, w, h = map(int, ann['bbox'])
        mask = coco.annToMask(ann)[y:y+h, x:x+w]
        mask = (mask > 0).astype(np.uint8)

        cropped_cube = cube[y:y+h, x:x+w, :]
        masked_cube = cropped_cube * mask[:, :, np.newaxis]

        # Save with file base and annotation ID
        filename = f"{file_base}_ann{ann_id}.npy"
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, masked_cube)
        print(f"🟩 Subcube saved: {output_path}")


def empty_output_dir(directory):
    if os.path.exists(directory):
        print(f"🧹 Clearing output directory: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory)


def process_image(image_id, coco, img_dir, out_dir):
    img_info = coco.loadImgs([image_id])[0]
    file_name = img_info['file_name']
    print(f"\n🔍 Processing image ID {image_id} → file name: {file_name}")

    image_path = find_image_recursive(img_dir, file_name)
    print(f"📂 Located image at: {image_path}")

    paths = find_hypercube_files_from_path(image_path)
    if not paths:
        raise FileNotFoundError("Missing HDR files")

    cube = load_corrected_hypercube(paths)
    print(f"✅ Corrected cube shape: {cube.shape}")

    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    if not anns:
        raise ValueError("No annotations found for this image")

    extract_cropped_cubes(cube, coco, img_info, anns, output_dir=out_dir)


def build_paths(class_name):
    json_path = os.path.join(ANNOTATIONS_BASE, class_name, f"{class_name}_annotations_coco.json")
    out_dir = os.path.join(OUTPUT_BASE, class_name)

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"❌ JSON file not found: {json_path}")

    return json_path, out_dir

"""
# Main function to execute the script for a specific class
def main():
    try:
        json_path, out_dir = build_paths(CLASS_NAME)
        coco = COCO(json_path)
        image_ids = coco.getImgIds()

        empty_output_dir(out_dir)

        for image_id in image_ids:
            process_image(image_id, coco, IMG_DIR, out_dir)

    except Exception as e:
        print(f"\n🛑 Fatal error: {e}")
        if 'out_dir' in locals():
            empty_output_dir(out_dir)
        sys.exit(1)
"""


def main():
    for class_id in range(4):  # C0 to C3
        class_name = f"C{class_id}"
        out_dir = None
        try:
            print(f"\n🚀 Processing class: {class_name}")
            json_path, out_dir = build_paths(class_name)
            coco = COCO(json_path)
            image_ids = coco.getImgIds()

            empty_output_dir(out_dir)

            for image_id in image_ids:
                process_image(image_id, coco, IMG_DIR, out_dir)

        except Exception as e:
            print(f"\n🛑 Fatal error while processing {class_name}: {e}")
            if out_dir is not None:
                empty_output_dir(out_dir)


if __name__ == '__main__':
    main()
