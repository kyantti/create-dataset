# Hypercube Visualization Script

This script (`visualize_hypercube.py`) provides comprehensive visualization tools for 3D hyperspectral data stored in `.npy` format.

## Features

The script offers multiple visualization modes:

1. **RGB Composite**: Creates a false-color RGB image from selected spectral bands
2. **Spectral Bands Grid**: Displays a grid of evenly-spaced spectral bands
3. **Mean Spectrum**: Plots the average spectral signature across the entire image
4. **Spectral Signature**: Shows the spectrum at a specific spatial location
5. **3D Visualization**: Creates a 3D surface plot of a selected band
6. **Interactive Band Viewer**: Browse through all spectral bands interactively with a slider

## Usage

### Basic Usage (Default File)

The script uses the default hypercube file if no path is provided:

```bash
python visualize_hypercube.py
```

This will generate all visualizations for the file:
`outputs/cropped-hypercubes/C0/Fx10_20230717_Riego_C0_2023-07-17_09-02-21_ann0.npy`

### Specify a Different File

```bash
python visualize_hypercube.py path/to/your/hypercube.npy
```

### Visualization Modes

#### 1. All Visualizations (Default)

Generate all visualization types:

```bash
python visualize_hypercube.py --mode all
```

#### 2. RGB Composite Only

```bash
python visualize_hypercube.py --mode rgb
```

#### 3. Spectral Bands Grid

```bash
python visualize_hypercube.py --mode bands
```

#### 4. Mean Spectrum Plot

```bash
python visualize_hypercube.py --mode spectrum
```

#### 5. Spectral Signature at a Point

```bash
# Center point (default)
python visualize_hypercube.py --mode signature

# Specific coordinates
python visualize_hypercube.py --mode signature --x 100 --y 80
```

#### 6. 3D Visualization

```bash
# Middle band (default)
python visualize_hypercube.py --mode 3d

# Specific band
python visualize_hypercube.py --mode 3d --band 35
```

#### 7. Interactive Band Viewer

```bash
python visualize_hypercube.py --mode interactive
```

This opens an interactive window with a slider to browse through all spectral bands.

### Save Visualizations

To save visualizations to a directory:

```bash
python visualize_hypercube.py --mode all --output-dir outputs/visualizations
```

This will create PNG files for each visualization in the specified directory.

## Examples

### Example 1: Quick RGB Preview

```bash
python visualize_hypercube.py --mode rgb
```

### Example 2: Analyze Specific Point

```bash
python visualize_hypercube.py --mode signature --x 224 --y 80 --output-dir outputs/analysis
```

### Example 3: Complete Analysis with Saved Outputs

```bash
python visualize_hypercube.py \
  outputs/cropped-hypercubes/C0/Fx10_20230717_Riego_C0_2023-07-17_09-02-21_ann0.npy \
  --mode all \
  --output-dir outputs/visualizations/ann0
```

### Example 4: Interactive Exploration

```bash
python visualize_hypercube.py --mode interactive
```

Use the slider to browse through all 70 spectral bands interactively.

## Command-Line Options

```
positional arguments:
  file_path             Path to the .npy hypercube file
                        (default: outputs/cropped-hypercubes/C0/Fx10_20230717_Riego_C0_2023-07-17_09-02-21_ann0.npy)

options:
  -h, --help            Show help message
  --mode {all,rgb,bands,spectrum,signature,3d,interactive}
                        Visualization mode (default: all)
  --output-dir OUTPUT_DIR
                        Directory to save output images
  --x X                 X coordinate for spectral signature plot
  --y Y                 Y coordinate for spectral signature plot
  --band BAND           Band index for 3D visualization
```

## Data Format

The script expects `.npy` files containing 3D numpy arrays with shape `(bands, height, width)`:
- **bands**: Number of spectral bands (e.g., 70)
- **height**: Image height in pixels (e.g., 161)
- **width**: Image width in pixels (e.g., 448)

## Output Files

When using `--output-dir`, the following files are generated (in `all` mode):

- `{filename}_rgb.png`: RGB composite image
- `{filename}_bands_grid.png`: Grid of spectral bands
- `{filename}_mean_spectrum.png`: Mean spectral signature plot
- `{filename}_spectral_signature.png`: Spectral signature at center point
- `{filename}_3d.png`: 3D visualization of selected band

## Dependencies

The script requires the following Python packages:
- `numpy`: For array operations
- `matplotlib`: For visualization
- `spectral`: For hyperspectral data handling (optional, imported for future enhancements)

Install dependencies:

```bash
pip install numpy matplotlib
```

## Tips

1. **Performance**: The 3D visualization downsamples the data for faster rendering. For very large hypercubes, consider using a specific band rather than browsing all bands interactively.

2. **RGB Band Selection**: The default RGB composite uses bands (50, 35, 20). You can modify these in the `create_rgb_composite()` function if you need different band combinations.

3. **Memory**: Loading very large hypercubes may require significant RAM. If you encounter memory issues, consider processing specific bands or regions of interest.

4. **Customization**: The script is designed to be easily customizable. You can modify band selections, color maps, and other parameters directly in the code.
