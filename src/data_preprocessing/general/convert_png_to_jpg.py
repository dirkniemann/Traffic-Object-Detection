"""
This script converts PNG images to JPG format in a specified directory using a GPU for faster processing.
"""
import os
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# List of folders to be processed
FOLDERS_TO_PROCESS = [
    "data/bdd100k/images/train"
]

# Batch size for processing (adjust to GPU memory)
BATCH_SIZE = 4160

def print_gpu_memory_usage():
    """Prints the current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"GPU memory allocated: {allocated:.2f} MB", flush=True)
    print(f"GPU memory reserved: {reserved:.2f} MB", flush=True)

def load_images(file_paths):
    """Loads images as NumPy arrays and returns them as a list."""
    images = []
    valid_paths = []
    for file_path in file_paths:
        try:
            # Load the image with OpenCV
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                images.append(img)
                valid_paths.append(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}", flush=True)
    return images, valid_paths


def save_images(images, file_paths):
    """Saves the converted images as JPG files and removes the original files."""
    for img, file_path in zip(images, file_paths):
        try:
            # Save the image as JPG in the same directory as the original PNG
            jpg_path = file_path.replace(".png", ".jpg")
            cv2.imwrite(jpg_path, img)
            
            # Delete the original file
            os.remove(file_path)
        except Exception as e:
            print(f"Error saving {file_path}: {e}", flush=True)


def process_batch(images, device):
    """Processes a batch of images on the GPU."""
    # Convert images to tensors and move them to the GPU
    tensor_images = [torch.tensor(img, device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0 for img in images]
    batch_tensor = torch.stack(tensor_images)  # Create a batch
    
    # Print GPU memory usage
    print_gpu_memory_usage()
    # Optional: Additional image processing operations could be added here.
    
    # Convert back to NumPy arrays for saving
    processed_images = (batch_tensor * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return processed_images

def process_folder(folder_path, device):
    """Recursively searches the folder and processes images in batches."""
    # Find all PNG files
    files = [
        os.path.join(root, file)
        for root, _, filenames in os.walk(folder_path)
        for file in filenames
        if file.lower().endswith(".png")
    ]
    
    print(f"{len(files)} files found in {folder_path}", flush=True)
    
    # Process files in batches
    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i + BATCH_SIZE]
        images, valid_paths = load_images(batch_files)
        
        if images:
            # Processing on the GPU
            processed_images = process_batch(images, device)
            
            # Save the converted images
            save_images(processed_images, valid_paths)
            print(f"Batch {i // BATCH_SIZE + 1} processed ({len(images)} images).", flush=True)

if __name__ == "__main__":
    # Check if a GPU is available
    if not torch.cuda.is_available():
        print("No GPU available! Please ensure an NVIDIA GPU is installed.", flush=True)
        exit(1)
    
    # Initialize GPU device
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}", flush=True)

    # Process all defined folders
    for folder in FOLDERS_TO_PROCESS:
        print(f"Processing folder: {folder}", flush=True)
        process_folder(folder, device)

    print("Processing completed.", flush=True)
