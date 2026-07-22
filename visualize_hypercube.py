import numpy as np
from spectral import view_cube
import os

# Path to the cube file
cube_path = 'outputs/cropped-hypercubes/C0/Fx10_20230717_Riego_C0_2023-07-17_09-02-21_ann0.npy'

# Load the cube
cube = np.load(cube_path)

print(f"Loaded cube shape: {cube.shape}")

# Visualize the cube (opens an interactive window)
# You can specify bands if you know which ones to use for RGB, e.g., bands=[29, 19, 9]
view_cube(cube)

print("pfr")

# To save a screenshot, you will need to do it manually from the window, as view_cube does not support direct saving.
# For saving an RGB image, use save_rgb from spectral:
# from spectral import save_rgb
# save_rgb('output_rgb.png', cube, [29, 19, 9])
