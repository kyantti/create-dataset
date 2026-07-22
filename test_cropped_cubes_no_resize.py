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
            print(f"❌ Error processing {file}: {e}")
            print("🛑 Stopping process due to error.")
            raise e

def process_all_cropped_hypercubes(input_base_dir='outputs/cropped-hypercubes', 
                                  output_base_dir='outputs/rgb-test-no-resize', 
                                  r=1, g=2, b=0):
    """Process all cropped hypercubes and create RGB test images"""
    # Create the base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all subdirectories (C0, C1, C2, C3, etc.)
    if not os.path.exists(input_base_dir):
        print(f"❌ Input directory {input_base_dir} does not exist!")
        return
    
    subdirs = [d for d in os.listdir(input_base_dir) 
               if os.path.isdir(os.path.join(input_base_dir, d))]
    subdirs.sort()  # Sort for consistent processing order
    
    print(f"📁 Found {len(subdirs)} subdirectories: {subdirs}")
    
    total_files_processed = 0
    
    for subdir in subdirs:
        input_dir = os.path.join(input_base_dir, subdir)
        output_dir = os.path.join(output_base_dir, subdir)
        
        print(f"\n📂 Processing {subdir}...")
        print(f"   Input:  {input_dir}")
        print(f"   Output: {output_dir}")
        
        # Count files in this subdirectory
        patch_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
        total_files_processed += len(patch_files)
        
        # Process this subdirectory
        visualize_all_patches(input_dir, output_dir, r, g, b)
    
    print("\n🎉 Processing complete!")
    print(f"📊 Total files processed: {total_files_processed}")
    print(f"📁 RGB test images saved to: {output_base_dir}")

if __name__ == '__main__':
    print("🚀 Starting RGB preview generation for all cropped hypercubes...")
    process_all_cropped_hypercubes()
