import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
model = YOLO('/home/dinesh/CV_Lehren/task_2_CV/yolo11x.pt')

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

def calculate_metrics(gt_boxes, pred_boxes, iou_threshold=0.50):
    """
    Calculate precision, recall, and IoU metrics for a single image.
    """
    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for pred in pred_boxes:
        best_iou = 0
        best_match = None
        for idx, gt in enumerate(gt_boxes):
            iou = calculate_iou(pred, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_match)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall, tp, fp, fn

def estimate_depth(bbox, intrinsic_matrix, plane_height=1.65):
    """
    Estimate the depth of an object using the intrinsic matrix and ground plane intersection.
    """
    x = (bbox[0] + bbox[2]) / 2  # Bottom-center x
    y = bbox[3]  # Bottom-center y
    point = np.array([x, y, 1])  # Homogeneous coordinates
    ray = np.linalg.inv(intrinsic_matrix) @ point
    t = plane_height / ray[1]  # Intersection with ground plane
    point_3d = t * ray
    distance = np.linalg.norm(point_3d)
    return distance

def annotate_with_background(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, text_color=(0, 0, 0), bg_color=(255, 255, 255), thickness=1):
    """
    Annotate text with a white background for better visibility.
    """
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    # Draw the white background rectangle
    cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), bg_color, -1)
    # Overlay the text
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)

def process_image(image_file, calib_file, label_file):
    """
    Process a single image: detect objects, calculate IoU, and estimate depth.
    Visualize predictions with labels (numbers for bounding boxes).
    Calculate precision and recall for the image.
    """
    # Load image
    image = cv2.imread(image_file)
    intrinsic_matrix = load_intrinsic_matrix(calib_file)
    gt_boxes = load_groundtruth_labels(label_file)

    # Detect objects using YOLO
    results = model.predict(image, conf=0.25)
    pred_boxes = []

    # Annotate YOLO bounding boxes with numbers
    for box_idx, box in enumerate(results[0].boxes):
        cls = int(box.cls.cpu().numpy()[0])  # Class index
        conf = float(box.conf.cpu().numpy()[0])  # Confidence score
        bbox = box.xyxy.cpu().numpy()[0]  # Bounding box in (xmin, ymin, xmax, ymax) format
        if cls == 2:  # Assuming 'car' class index is 2
            depth = estimate_depth(bbox, intrinsic_matrix)
            pred_boxes.append((bbox, conf, depth))  # Add depth to prediction data
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red box
            label = f"{box_idx + 1}"  # Label bounding boxes with numbers
            annotate_with_background(image, label, (x_min, y_min - 10), text_color=(0, 0, 0), bg_color=(255, 255, 255))

    # Annotate GT bounding boxes with numbers
    for gt_idx, gt in enumerate(gt_boxes):
        x_min, y_min, x_max, y_max = map(int, gt["bbox"])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box
        label = f"{gt_idx + 1}"  # Label ground truth boxes with numbers
        annotate_with_background(image, label, (x_min, y_min - 10), text_color=(0, 0, 0), bg_color=(255, 255, 255))

    # Calculate precision and recall
    precision, recall, tp, fp, fn = calculate_metrics(gt_boxes, [box[0] for box in pred_boxes])

    # Annotate precision and recall on the top-left corner with a white background
    precision_recall_text = [
        f"Precision: {precision:.2f}",
        f"Recall: {recall:.2f}"
    ]
    for idx, text in enumerate(precision_recall_text):
        annotate_with_background(image, text, (10, 30 + idx * 30), font_scale=0.8, thickness=2)

    # Log IoU, YOLO distances, and GT distances
    print(f"\nImage: {os.path.basename(image_file)} | Precision: {precision:.2f}, Recall: {recall:.2f}")
    for pred_idx, (pred_bbox, _, yolo_distance) in enumerate(pred_boxes):
        for gt_idx, gt in enumerate(gt_boxes):
            iou = calculate_iou(pred_bbox, gt["bbox"])
            gt_distance = gt["distance"]
            print(
                f"Image {os.path.basename(image_file)}, BB {pred_idx + 1} (YOLO: {yolo_distance:.2f}m) "
                f"vs GT {gt_idx + 1} (GT: {gt_distance:.2f}m): IoU={iou:.2f}"
            )

    # Save the annotated image
    output_file = os.path.join(OUTPUT_DIR, os.path.basename(image_file))
    cv2.imwrite(output_file, image)
    print(f"Saved annotated image to: {output_file}")

    return precision, recall, pred_boxes, gt_boxes

def plot_distance_comparison(pred_distances, gt_distances):
    """
    Generate a scatter plot comparing YOLO-predicted distances and ground truth distances.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(pred_distances, gt_distances, color='blue', label='Data points')
    plt.plot([0, max(pred_distances + gt_distances)], [0, max(pred_distances + gt_distances)],
             color='black', linestyle='--', label='Ideal line')

    plt.xlabel('Distance calculated using camera information', fontsize=12)
    plt.ylabel('Distance provided in ground truth', fontsize=12)
    plt.title('Comparison of Estimated and Ground Truth Distances', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.savefig('distance_comparison.png', dpi=300)
    plt.show()

def main():
    total_tp, total_fp, total_fn = 0, 0, 0
    all_precisions = []
    all_recalls = []
    all_pred_distances = []
    all_gt_distances = []

    for image_idx, image_name in enumerate(os.listdir(IMAGES_DIR)):
        if not image_name.endswith(".png"):
            continue

        # Paths
        image_file = os.path.join(IMAGES_DIR, image_name)
        label_file = os.path.join(LABELS_DIR, os.path.splitext(image_name)[0] + ".txt")
        calib_file = os.path.join(CALIB_DIR, os.path.splitext(image_name)[0] + ".txt")

        print(f"Processing Image {image_idx + 1}: {image_name}")

        # Process image
        precision, recall, pred_boxes, gt_boxes = process_image(image_file, calib_file, label_file)

        # Log metrics
        all_precisions.append(precision)
        all_recalls.append(recall)

        # Match predicted and ground truth distances using IoU
        for pred_bbox, _, pred_distance in pred_boxes:
            best_iou = 0
            best_gt_distance = None
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_bbox, gt_box["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_distance = gt_box["distance"]

            # Append distances if IoU is above the threshold
            if best_iou >= 0.5:  # Example IoU threshold
                all_pred_distances.append(pred_distance)
                all_gt_distances.append(best_gt_distance)

    # Calculate and log overall metrics
    mean_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    print(f"\nOverall Precision: {mean_precision:.2f}")
    print(f"Overall Recall: {mean_recall:.2f}")

    # Generate distance comparison plot
    if all_pred_distances and all_gt_distances:
        plot_distance_comparison(all_pred_distances, all_gt_distances)
    else:
        print("No valid distances to plot.")


if __name__ == "__main__":
    main()
