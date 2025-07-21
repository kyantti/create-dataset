import os
import numpy as np
from PIL import Image

def create_rgb_from_patch(patch, r=60, g=30, b=10):
    rgb = patch[:, :, [r, g, b]]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb

def visualize_all_patches(patch_dir, output_dir='rgb_previews', r=60, g=30, b=10):
    os.makedirs(output_dir, exist_ok=True)
    patch_files = [f for f in os.listdir(patch_dir) if f.endswith('.npy')]
    print(f"🔎 Found {len(patch_files)} .npy files in {patch_dir}")

    for file in patch_files:
        patch_path = os.path.join(patch_dir, file)
        patch = np.load(patch_path)
        try:
            rgb = create_rgb_from_patch(patch, r, g, b)
            img = Image.fromarray(rgb)
            output_name = os.path.splitext(file)[0] + '_rgb.png'
            img.save(os.path.join(output_dir, output_name))
            print(f"✅ Saved: {output_name}")
        except Exception as e:
            print(f"⚠️ Skipped {file}: {e}")

if __name__ == '__main__':
    print("🚀 Starting RGB preview generation...")
    visualize_all_patches('./cropped_cubes/C0/')
