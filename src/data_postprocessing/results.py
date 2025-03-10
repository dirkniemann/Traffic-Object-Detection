import os
import numpy as np
import matplotlib.pyplot as plt
import csv

# Directory containing the txt files
directory = 'runs/track/Video_VoloV11m_20241228'
labels_path = os.path.join(directory, 'labels')
csv_filename = os.path.join(directory, 'statistics.csv')

# Mapping of indices to labels
labels = [
    "pedestrian", "car", "truck", "bicycle", "traffic light", "speed limit 20 (prohibitory)",
    "speed limit 30 (prohibitory)", "speed limit 50 (prohibitory)", "speed limit 60 (prohibitory)",
    "speed limit 70 (prohibitory)", "speed limit 80 (prohibitory)", "restriction ends 80 (other)",
    "speed limit 100 (prohibitory)", "speed limit 120 (prohibitory)", "no overtaking (prohibitory)",
    "no overtaking (trucks) (prohibitory)", "priority at next intersection (danger)", "priority road (other)",
    "give way (other)", "stop (other)", "no traffic both ways (prohibitory)", "no trucks (prohibitory)",
    "no entry (other)", "danger (danger)", "bend left (danger)", "bend right (danger)", "bend (danger)",
    "uneven road (danger)", "slippery road (danger)", "road narrows (danger)", "construction (danger)",
    "traffic signal (danger)", "pedestrian crossing (danger)", "school crossing (danger)", "cycles crossing (danger)",
    "snow (danger)", "animals (danger)", "restriction ends (other)", "go right (mandatory)", "go left (mandatory)",
    "go straight (mandatory)", "go right or straight (mandatory)", "go left or straight (mandatory)",
    "keep right (mandatory)", "keep left (mandatory)", "roundabout (mandatory)", "restriction ends (overtaking) (other)",
    "restriction ends (overtaking (trucks)) (other)"
]

# Initialize statistics dictionary
stats = {label: {'count': 0, 'total_confidence': 0.0} for label in labels}

# Iterate over all txt files in the labels path
for filename in os.listdir(labels_path):
    if filename.endswith('.txt'):
        with open(os.path.join(labels_path, filename), 'r') as file:
            for line in file:
                parts = line.strip().split()
                label_index = int(parts[0])
                confidence = float(parts[5])
                label = labels[label_index]
                stats[label]['count'] += 1
                stats[label]['total_confidence'] += confidence

# Calculate average confidence for each label
for label in stats:
    if stats[label]['count'] > 0:
        stats[label]['average_confidence'] = stats[label]['total_confidence'] / stats[label]['count']
    else:
        stats[label]['average_confidence'] = 0.0

# Print statistics
for label, data in stats.items():
    print(f"{label}: Count = {data['count']}, Average Confidence = {data['average_confidence']:.2f}")

# Write statistics to CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Label', 'Count', 'Average Confidence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for label, data in stats.items():
        writer.writerow({'Label': label, 'Count': data['count'], 'Average Confidence': data['average_confidence']})

# delete labels labels path
os.system(f'rm -r {labels_path}')