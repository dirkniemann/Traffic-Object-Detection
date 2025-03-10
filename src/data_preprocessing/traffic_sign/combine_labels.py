"""
This script prepares images from the GTSDB dataset for the final model. It combines the labels from the dataset and adjusts the indices. 
Additionally, it adds labels from the detected objects of the BDD100k model and removes the split between train and val.
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

def remove_split(sub_folder):
    train_folder = os.path.join(sub_folder, 'train')
    val_folder = os.path.join(sub_folder, 'val')

    for folder in [train_folder, val_folder]:
        for file_name in os.listdir(folder):
            if file_name.endswith(('.txt', '.jpg')):
                src_path = os.path.join(folder, file_name)
                dst_path = os.path.join(sub_folder, file_name)
                os.rename(src_path, dst_path)

if __name__ == "__main__":

    main_folder = 'data/traffic_signs'
    label_folder = os.path.join(main_folder, 'labels')
    images_folder = os.path.join(main_folder, 'images')
    # Remove all .cache files in the label_folder
    for file_name in os.listdir(label_folder):
        if file_name.endswith('.cache'):
            os.remove(os.path.join(label_folder, file_name))
    remove_split(label_folder)
    remove_split(images_folder)
    
    # Load bounding boxes from the dataset
    for file_name in os.listdir(label_folder):
        object_boxes = []
        dataset_boxes = []
        if file_name.endswith('.txt'):
            if os.path.exists(os.path.join(label_folder, "object_detection", file_name)):
                object_boxes = load_bounding_boxes(os.path.join(label_folder, "object_detection", file_name))
            else:
                print(f"No object detection file found for {file_name}")
            dataset_boxes = load_bounding_boxes(os.path.join(label_folder, file_name))
            #Increase the index of the dataset boxes by 5
            for box in dataset_boxes:
                box[0] = str(int(box[0]) + 5)
            combined_boxes = object_boxes + dataset_boxes
            save_combined_boxes(os.path.join(label_folder, file_name), combined_boxes)
            