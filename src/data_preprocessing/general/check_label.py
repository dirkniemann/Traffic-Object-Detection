import os
import csv
from collections import Counter

import matplotlib.pyplot as plt

# Define the path to the directory containing the txt files
directory_path = 'data/finetuning/labels/train'

# Define the label mapping
label_mapping = {
    0: 'pedestrian',
    1: 'car',
    2: 'truck',
    3: 'bicycle',
    4: 'traffic light',
    5: 'speed limit 20 (prohibitory)',
    6: 'speed limit 30 (prohibitory)',
    7: 'speed limit 50 (prohibitory)',
    8: 'speed limit 60 (prohibitory)',
    9: 'speed limit 70 (prohibitory)',
    10: 'speed limit 80 (prohibitory)',
    11: 'restriction ends 80 (other)',
    12: 'speed limit 100 (prohibitory)',
    13: 'speed limit 120 (prohibitory)',
    14: 'no overtaking (prohibitory)',
    15: 'no overtaking (trucks) (prohibitory)',
    16: 'priority at next intersection (danger)',
    17: 'priority road (other)',
    18: 'give way (other)',
    19: 'stop (other)',
    20: 'no traffic both ways (prohibitory)',
    21: 'no trucks (prohibitory)',
    22: 'no entry (other)',
    23: 'danger (danger)',
    24: 'bend left (danger)',
    25: 'bend right (danger)',
    26: 'bend (danger)',
    27: 'uneven road (danger)',
    28: 'slippery road (danger)',
    29: 'road narrows (danger)',
    30: 'construction (danger)',
    31: 'traffic signal (danger)',
    32: 'pedestrian crossing (danger)',
    33: 'school crossing (danger)',
    34: 'cycles crossing (danger)',
    35: 'snow (danger)',
    36: 'animals (danger)',
    37: 'restriction ends (other)',
    38: 'go right (mandatory)',
    39: 'go left (mandatory)',
    40: 'go straight (mandatory)',
    41: 'go right or straight (mandatory)',
    42: 'go left or straight (mandatory)',
    43: 'keep right (mandatory)',
    44: 'keep left (mandatory)',
    45: 'roundabout (mandatory)',
    46: 'restriction ends (overtaking) (other)',
    47: 'restriction ends (overtaking (trucks)) (other)'
}

# Initialize a counter for the labels
label_counter = Counter()

# Iterate through all txt files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        with open(os.path.join(directory_path, filename), 'r') as file:
            for line in file:
                label_index = int(line.split()[0])
                label_counter[label_index] += 1

# Convert the counter to a dictionary with label names
label_counts = {k: (label_mapping[k], v) for k, v in label_counter.items()}

# Save the counts to a CSV file
csv_file_path = 'label_counts_finetuning.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'Name', 'Counter'])
    for index in sorted(label_counts.keys()):
        writer.writerow([index, label_counts[index][0], label_counts[index][1]])

# Plot the counts as a bar chart
plt.figure(figsize=(12, 8))
sorted_label_counts = dict(sorted(label_counts.items()))
labels = [label_mapping[index] for index in sorted_label_counts.keys()]
counts = [count for _, count in sorted_label_counts.values()]

plt.bar(labels, counts)
plt.xticks(rotation=90)
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Label Counts in Dataset')
plt.tight_layout()
plt.savefig('label_counts_finetuning.png')
plt.show()