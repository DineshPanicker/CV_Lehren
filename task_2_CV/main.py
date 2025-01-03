import os
import cv2
import numpy as np
from ultralytics import YOLO

# Paths to your directories
CALIB_DIR = '/home/dinesh/KITTI_Selection/KITTI_Selection/calib'
IMAGES_DIR = '/home/dinesh/KITTI_Selection/KITTI_Selection/images'
LABELS_DIR = '/home/dinesh/KITTI_Selection/KITTI_Selection/labels'
OUTPUT_DIR = '/home/dinesh/KITTI_Selection/KITTI_Selection/output' 

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Output directory created: {OUTPUT_DIR}")
else:
    print(f"Output directory already exists: {OUTPUT_DIR}")

# Initialize YOLO object detection model
model = YOLO(r'/home/dinesh/CV_Lehren/task_2_CV/yolo11n.pt')  # Replace "yolov11" with the appropriate model path if needed


def load_intrinsic_matrix(calib_file):
    """
    Load the intrinsic camera matrix from the calibration file.
    """
    with open(calib_file, "r") as f:
        lines = f.readlines()
        matrix = [list(map(float, line.strip().split())) for line in lines]
    return np.array(matrix)


def load_groundtruth_labels(label_file):
    """
    Load ground truth bounding boxes and distances from a label file.
    """
    gt_boxes = []
    with open(label_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            obj_type, xmin, ymin, xmax, ymax, gt_distance = parts
            gt_boxes.append({
                "type": obj_type,
                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "distance": float(gt_distance)
            })
    return gt_boxes


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area_box1 + area_box2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(gt_boxes, pred_boxes, iou_threshold=0.90):
    """
    Calculate precision, recall, and IoU metrics for a single image.
    """
    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for pred in pred_boxes:
        best_iou = 0
        for idx, gt in enumerate(gt_boxes):
            iou = calculate_iou(pred, gt["bbox"])
            if iou > best_iou:
                best_iou = iou

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall, tp, fp, fn


def estimate_depth(bbox, intrinsic_matrix, camera_height=1.65):
    """
    Estimate the depth of an object using the intrinsic camera matrix and bounding box coordinates.
    """
    x, y, w, h = bbox
    midpoint = np.array([x + w / 2, y + h, 1])
    world_coords = np.linalg.inv(intrinsic_matrix).dot(midpoint)
    depth = camera_height / world_coords[1]
    return depth


def process_image(image_file, calib_file, label_file):
    """
    Process a single image: detect objects, calculate IoU, and estimate depth.
    Save processed images with bounding box annotations to the output folder.
    """
    # Extract image name
    image_name = os.path.basename(image_file)
    print(f"\nProcessing Image: {image_name}")

    # Load image and intrinsic matrix
    image = cv2.imread(image_file)
    intrinsic_matrix = load_intrinsic_matrix(calib_file)

    # Load ground truth labels
    gt_boxes = load_groundtruth_labels(label_file)

    # Detect objects using YOLO
    results = model.predict(image, conf=0.3)

    # Extract bounding boxes and classes
    pred_boxes = []
    for box in results[0].boxes:  # Access boxes from the first result
        cls = int(box.cls.cpu().numpy()[0])  # Class index
        conf = float(box.conf.cpu().numpy()[0])  # Confidence score
        bbox = box.xyxy.cpu().numpy()[0]  # Bounding box in (xmin, ymin, xmax, ymax) format

        if cls == 2:  # Assuming 'car' class index is 2
            pred_boxes.append((bbox, conf))
            print(f"  Predicted BB: {bbox}, Confidence: {conf:.2f}")

            # Draw YOLO predicted bounding box in RED
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red color
            label = f"Pred: {conf:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw ground truth bounding boxes in GREEN
    print("  Ground Truth BBs:")
    for gt in gt_boxes:
        x_min, y_min, x_max, y_max = map(int, gt["bbox"])
        print(f"    BB: {gt['bbox']}, Distance: {gt['distance']:.2f}m")
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green color
        label = f"GT: {gt['distance']:.2f}m"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image to the output directory
    output_file = os.path.join(OUTPUT_DIR, image_name)
    cv2.imwrite(output_file, image)
    print(f"Saved annotated image to: {output_file}")

    return pred_boxes  # Return predictions for further evaluation


def main():
    total_tp, total_fp, total_fn = 0, 0, 0
    all_precisions = []
    all_recalls = []
    depth_errors = []

    for image_name in os.listdir(IMAGES_DIR):
        if not image_name.endswith(".png"):
            continue

        # Derive file paths
        image_file = os.path.join(IMAGES_DIR, image_name)
        label_file = os.path.join(LABELS_DIR, os.path.splitext(image_name)[0] + ".txt")
        calib_file = os.path.join(CALIB_DIR, os.path.splitext(image_name)[0] + ".txt")

        # Process image
        pred_boxes = process_image(image_file, calib_file, label_file)

        # Extract ground truth
        gt_boxes = load_groundtruth_labels(label_file)

        # Calculate metrics
        precision, recall, tp, fp, fn = calculate_metrics(gt_boxes, [box[0] for box in pred_boxes])
        all_precisions.append(precision)
        all_recalls.append(recall)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Overall metrics
    mean_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0

    print(f"Overall Precision: {mean_precision:.2f}")
    print(f"Overall Recall: {mean_recall:.2f}")


if __name__ == "__main__":
    main()
