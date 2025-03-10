"""
This script renames label files in a specified directory by replacing "_label_" with "_camera_" in the filenames.
"""
import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if "_label_frontcenter_" in filename:
            new_filename = filename.replace("_label_", "_camera_")
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} to {new_file}')

if __name__ == "__main__":
    directory = "data/audi_a2d2/labels"  # Replace with your directory path
    rename_files(directory)