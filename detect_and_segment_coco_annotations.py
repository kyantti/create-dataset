import os
import sys
import os
import cv2
import json
import torch
import numpy as np
from supervision.annotators.core import BoxAnnotator, LabelAnnotator, MaskAnnotator
from supervision.detection.core import Detections
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from datetime import datetime
from file_processing import process_images, get_images_by_class, parse_filename

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Grounded-SAM-2")))
try:
    from sam2.build_sam import build_sam2 # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor # type: ignore
    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict # type: ignore
    print("Importing from Grounded-SAM-2 succeeded")
except ImportError as e:
    print("Importing from Grounded-SAM-2 failed")
    raise e

"""
Hyper parameters
"""
TEXT_PROMPT = "fig."
IMAGES_DIR = "/home/pablosetra/Pictures/imagenes-higos/raw/julio"
SAM2_CHECKPOINT = "./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "./Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT = "./Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
MAX_BOX_WIDTH = 250
MAX_BOX_HEIGHT = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("./outputs/segm-masks-2023-july-17-28")
DUMP_JSON_RESULTS = True

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Global variables for COCO format
coco_data = {
    "info": {
        "year": str(datetime.now().year),
        "version": "1.0",
        "description": "Fig detection dataset using SAM2 and Grounding DINO",
        "contributor": "Pablo Setrakian Bearzotti",
        "url": "",
        "date_created": datetime.now().isoformat()
    },
    "licenses": [
        {
            "id": 1,
            "url": "NO URL",
            "name": "NO NAME"
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "fig",
            "supercategory": "fruit"
        }
    ],
    "images": [],
    "annotations": []
}

# Global counters
image_id_counter = 0
annotation_id_counter = 0

# Build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)


# Build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


def load_and_predict(img_path, text, sam2_predictor, grounding_model):
    """
    Load image and perform prediction using SAM2 and Grounding DINO.
    
    Args:
        img_path (str): Path to the input image.
        text (str): Text prompt for grounding.
        sam2_predictor (SAM2ImagePredictor): SAM2 image predictor instance.
        grounding_model: Grounding DINO model instance.
    
    Returns:
        input_boxes (np.ndarray): Bounding boxes for detected objects.
        masks (np.ndarray): Masks for detected objects.
        class_ids (np.ndarray): Class IDs for detected objects.
        scores (np.ndarray): Confidence scores for the detections.
        labels (list): Labels for the detected objects.
        w (int): Width of the input image.
        h (int): Height of the input image.
        class_names (list): Names of the classes detected.
    """
    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)

    # Grounding DINO prediction (NO autocast - like original)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    # Process boxes and filter by size (like in your original script)
    h, w, _ = image_source.shape
        
    # Convert normalized (cx, cy, w, h) to pixel coordinates
    boxes_pixel = boxes * torch.tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy")
        
    # Calculate width and height for filtering
    widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        
    # Filter boxes by size (configurable parameters)
    size_mask = (widths <= MAX_BOX_WIDTH) & (heights <= MAX_BOX_HEIGHT)
        
    # Apply the mask to filter boxes, confidences, and labels
    boxes = boxes[size_mask]
    confidences = confidences[size_mask]
    labels = [label for i, label in enumerate(labels) if size_mask[i]]
           
    # Convert filtered boxes for SAM2
    boxes_pixel_filtered = boxes * torch.tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes_pixel_filtered, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # Configure TF32
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # NOW start autocast (like original) - ONLY for SAM2
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    return input_boxes, masks, class_ids, scores, labels, w, h, class_names


def visualize_results(img_path, input_boxes, masks, class_ids, labels, output_path):
    """
    Visualize the results using supervision library.
    
    Args:
        img_path: Path to the input image.
        input_boxes: Bounding boxes for detected objects.
        masks: Masks for detected objects.
        class_ids: Class IDs for detected objects.
        labels: Labels for the detected objects.
        output_path: Directory to save the annotated images.
    """
    img = cv2.imread(img_path)
    detections = Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    img_name = Path(img_path).stem

    box_annotator = BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_path, f"{img_name}_annotated.jpg"), annotated_frame)
    mask_annotator = MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_path, f"{img_name}_annotated_with_mask.jpg"), annotated_frame)    


def single_mask_to_rle(mask):
    """
    Convert a single mask to RLE format for saving in JSON.
    """
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]  # type: ignore
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def bbox_xyxy_to_xywh(bbox):
    """
    Convert bounding box from (x1, y1, x2, y2) to (x, y, width, height) format.
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

def calculate_area(bbox):
    """
    Calculate area of bounding box in xywh format.
    """
    x, y, w, h = bbox
    return w * h

def dump_and_save_json(masks, input_boxes, scores, img_path, class_names, w, h, coco_data, image_id_counter, annotation_id_counter):
    """
    Add image and annotations to the COCO data structure.
    Returns updated image_id_counter and annotation_id_counter.
    Args:
        masks (np.ndarray): Masks for detected objects.
        input_boxes (np.ndarray): Bounding boxes for detected objects.
        scores (np.ndarray): Confidence scores for the detections.
        img_path (str): Path to the input image.
        class_names (list): Names of the classes detected.
        w (int): Width of the input image.
        h (int): Height of the input image.
    """
    if DUMP_JSON_RESULTS:
        # Add image information
        img_name = Path(img_path).name
        image_info = {
            "id": image_id_counter,
            "license": 1,
            "file_name": img_name,
            "height": h,
            "width": w,
            "date_captured": parse_filename(img_path).date
        }
        coco_data["images"].append(image_info)
        
        # Add annotations for each detection
        for i, (mask, bbox, score, class_name) in enumerate(zip(masks, input_boxes, scores, class_names)):
            # Convert mask to RLE format
            mask_rle = single_mask_to_rle(mask)
            
            # Convert bbox from xyxy to xywh format
            bbox_xywh = bbox_xyxy_to_xywh(bbox.tolist())
            
            # Calculate area
            area = calculate_area(bbox_xywh)
            
            # Create annotation
            annotation = {
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": 1,  # Assuming all detections are "fig" category
                "bbox": bbox_xywh,
                "area": area,
                "segmentation": mask_rle,
                "iscrowd": 0
            }
            
            coco_data["annotations"].append(annotation)
            annotation_id_counter += 1
        
        image_id_counter += 1
    return image_id_counter, annotation_id_counter


def save_final_coco_json(output_dir, class_id, coco_data):
    """
    Save the final COCO format JSON file.
    """
    coco_json_path = output_dir / f"C{class_id}_annotations_coco.json"
    with open(coco_json_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"COCO format annotations saved to: {coco_json_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")


 
def main():
    paths_by_class = get_images_by_class(process_images(IMAGES_DIR, verbose=False))
    total_masks = 0
    for class_id, paths in paths_by_class.items():
        # Initialize COCO data and counters per class
        coco_data = {
            "info": {
                "year": str(datetime.now().year),
                "version": "1.0",
                "description": "Fig detection dataset using SAM2 and Grounding DINO",
                "contributor": "Pablo Setrakian Bearzotti",
                "url": "",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "url": "NO URL",
                    "name": "NO NAME"
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "fig",
                    "supercategory": "fruit"
                }
            ],
            "images": [],
            "annotations": []
        }
        image_id_counter = 0
        annotation_id_counter = 0

        output_path = Path(OUTPUT_DIR) / f"C{class_id}"
        output_path.mkdir(parents=True, exist_ok=True)
        for img_path in paths:
            input_boxes, masks, class_ids, scores, labels, w, h, class_names = load_and_predict(
                img_path=img_path,
                text=TEXT_PROMPT,
                sam2_predictor=sam2_predictor,
                grounding_model=grounding_model
            )
            total_masks += len(masks)
            visualize_results(img_path, input_boxes, masks, class_ids, labels, output_path)
            image_id_counter, annotation_id_counter = dump_and_save_json(
                masks, input_boxes, scores, img_path, class_names, w, h,
                coco_data, image_id_counter, annotation_id_counter
            )
        save_final_coco_json(output_path, class_id, coco_data)
    print(f"Total masks generated: {total_masks}")


if __name__ == "__main__":  
    main()