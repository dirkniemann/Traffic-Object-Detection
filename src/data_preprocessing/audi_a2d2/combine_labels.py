"""
This script loads the detected bounding boxes from extract_bounding_boxes.py, which were created based on the semantic segmentation of the dataset. 
It checks if a bounding box with index 5 (Traffic Sign) is present. If so, it verifies whether a bounding box for the image was also found based on detection by the Yolo model and if the boxes overlap. 
If they do, the bounding box for the corresponding traffic sign is set, and the combined boxes are extracted again.
"""
import os

def load_bounding_boxes(file_path):
    # Load bounding boxes from a file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        boxes = [line.strip().split() for line in lines]
    return boxes

def save_combined_boxes(file_path, boxes):
    # Save combined bounding boxes to a file
    with open(file_path, 'w') as file:
        for box in boxes:
            file.write(f"{' '.join(box)}\n")

def iou(box1, box2):
    # Calculate Intersection over Union (IoU) of two bounding boxes
    x1, y1, w1, h1 = map(float, box1[:4])
    x2, y2, w2, h2 = map(float, box2[:4])
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def combine_labels(semantic_folder, signs_folder, output_folder, iou_threshold=0.4):
    # Combine labels from semantic and traffic sign folders based on IoU threshold
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for semantic_file in os.listdir(semantic_folder):
        semantic_file_path = os.path.join(semantic_folder, semantic_file)
        sign_file_path = os.path.join(signs_folder, semantic_file)
        combined_boxes = []
        semantic_boxes = load_bounding_boxes(semantic_file_path)
        for semantic_box in semantic_boxes:
            if semantic_box[0] == '5' and os.path.exists(sign_file_path):
                # Load traffic sign bounding boxes and check IoU
                sign_boxes = load_bounding_boxes(sign_file_path)
                for sign_box in sign_boxes:
                    if iou(semantic_box[1:], sign_box[1:]) > iou_threshold:
                        semantic_box[0] = sign_box[0]
                        combined_boxes.append(semantic_box)
                        break
            elif semantic_box[0] == '5':
                print(f"Warning: No traffic sign file found for {semantic_file}")
            else:
                combined_boxes.append(semantic_box)
        output_file_path = os.path.join(output_folder, semantic_file)
        save_combined_boxes(output_file_path, combined_boxes)

if __name__ == "__main__":
    # Define paths and IoU threshold
    semantic_folder = 'data/audi_a2d2/labels/labels_semantic'
    signs_folder = 'data/audi_a2d2/labels/labels_traffic_signs'
    output_folder = 'data/audi_a2d2/labels'
    
    # Combine labels
    combine_labels(semantic_folder, signs_folder, output_folder)