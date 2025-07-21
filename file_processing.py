from dataclasses import dataclass
import os
from datetime import datetime
import glob
from typing import List, Dict
from collections import Counter, defaultdict

@dataclass
class ImageInfo:
    """Represents information extracted from image filename."""

    path: str
    filename: str
    class_id: int
    date: str
    day: str


def find_png_images(base_path: str, verbose: bool = False) -> List[str]:
    """Find all PNG images in the dataset directory structure."""
    pattern = os.path.join(base_path, "**", "*.png")
    png_files = glob.glob(pattern, recursive=True)
    if not png_files:
        raise FileNotFoundError(
            f"No PNG images found in '{base_path}'. Please check the path."
        )
    if verbose:
        print(f"Found {len(png_files)} PNG images.")
    return sorted(png_files)


def get_day_from_date(date_str: str) -> str:
    """Extract day of week from date string."""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    day_names = [
        "lunes",
        "martes",
        "miércoles",
        "jueves",
        "viernes",
        "sábado",
        "domingo",
    ]
    return day_names[date_obj.weekday()]


def parse_filename(img_path: str, verbose: bool = False) -> ImageInfo:
    """Parse image filename to extract metadata."""
    filename = os.path.basename(img_path)
    base_name = filename.replace(".png", "")
    parts = base_name.split("_")
    class_id = None
    date_part = None
    day = None

    if verbose:
        print(f"  Parsing: {filename}")
        print(f"  Parts: {parts}")

    if len(parts) < 6:
        raise ValueError(
            f"Unexpected filename format: '{filename}' - expected at least 6 parts separated by underscores"
        )

    class_part = parts[3]  # Should be C0, C1, C2, C3
    date_part = parts[4]  # YYYY-MM-DD

    if verbose:
        print(f"  Class part: '{class_part}'")

    # Parse class ID
    if not class_part.startswith("C") or len(class_part) < 2:
        raise ValueError(
            f"Invalid class part: '{class_part}' - expected format 'C[0-9]+'"
        )

    class_suffix = class_part[1:]
    # Handle common OCR/typo issues: O->0, I->1, etc.
    class_suffix = class_suffix.replace("O", "0").replace("o", "0")
    class_suffix = class_suffix.replace("I", "1").replace("l", "1")

    try:
        class_id = int(class_suffix)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse class ID from '{class_suffix}' in filename '{filename}'"
        ) from exc

    # Parse date and extract day
    try:
        day = get_day_from_date(date_part)
    except ValueError as e:
        raise ValueError(
            f"Could not parse date '{date_part}' from filename '{filename}': {e}"
        ) from e

    if verbose:
        print(f"  Extracted: Class={class_id}, Day={day}, Date={date_part}")

    return ImageInfo(
        path=img_path,
        filename=filename,
        class_id=class_id,
        date=date_part,
        day=day,
    )


def process_images(dataset_path: str, verbose: bool = False) -> List[ImageInfo]:
    """
    Process all PNG images in the dataset directory and extract metadata.

    Args:
        dataset_path: Path to the dataset directory containing images
        verbose: If True, print detailed processing information
    Returns:
        List of ImageInfo objects containing metadata for each image
    """
    png_files = find_png_images(dataset_path, verbose)
    image_info_list = []

    for img_path in png_files:
        try:
            image_info = parse_filename(img_path, verbose)
            image_info_list.append(image_info)
        except ValueError as e:
            print(f"Error processing '{img_path}': {e}")

    if not image_info_list:
        raise ValueError("No valid images found after processing.")
    
    # Order the list by class_id and date
    image_info_list.sort(key=lambda x: (x.class_id, x.date))
    
    if verbose:
        display_dataset_summary(image_info_list)

    return image_info_list


def get_images_by_class(image_info_list: List[ImageInfo]) -> Dict[int, List[str]]:
    """
    Organize image paths by class ID.
    
    Args:
        image_info_list: List of ImageInfo objects
        
    Returns:
        Dictionary with class IDs as keys and lists of image paths as values
    """
    class_dict = defaultdict(list)
    for img_info in image_info_list:
        class_dict[img_info.class_id].append(img_info.path)
    
    # Ensure each class's images are sorted consistently
    for paths in class_dict.values():
        paths.sort()
    
    return dict(class_dict)


def display_dataset_summary(image_info_list: List[ImageInfo]) -> None:
    """
    Display summary statistics about the dataset.

    Args:
        image_info_list: List of ImageInfo objects containing metadata for each image

    Displays:
        - Total number of images
        - Count of images per class
        - Count of images per date
        - Detailed list of images per class  
    """
    # Count statistics
    total_images = len(image_info_list)
    class_counts = Counter(img.class_id for img in image_info_list)
    date_counts = Counter(img.date for img in image_info_list)
    
    # Group images by class for filename display
    images_by_class: Dict[int, List[str]] = {}
    for img in image_info_list:
        if img.class_id not in images_by_class:
            images_by_class[img.class_id] = []
        images_by_class[img.class_id].append(img.filename)
    
    # Display summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total images processed: {total_images}")
    
    print("\nImages per class:")
    for class_id in sorted(class_counts.keys()):
        print(f"  Class {class_id}: {class_counts[class_id]} images")
    
    print("\nImages per date:")
    for date in sorted(date_counts.keys()):
        print(f"  {date}: {date_counts[date]} images")
    
    print("\nDetailed class information:")
    for class_id in sorted(images_by_class.keys()):
        filenames = images_by_class[class_id]
        print(f"\nClass {class_id} ({len(filenames)} images):")
        for filename in sorted(filenames):
            print(f"  - {filename}")


if __name__ == "__main__":
    # Example usage
    EXAMPLE_PATH = "/home/pablosetra/Documents/higos"
    try:
        images = process_images(EXAMPLE_PATH, verbose=True)
        #result = get_images_by_class(process_images(EXAMPLE_PATH, verbose=True))
        #print(result)
        print(images)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")