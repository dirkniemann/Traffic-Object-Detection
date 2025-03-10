import pandas as pd
import os
import matplotlib.pyplot as plt

# Load data from CSV files
csv_YoloV11m = 'runs/detect/Video_VoloV11m/statistics.csv'
csv_YoloV11n = 'runs/track/Video_VoloV11m/statistics.csv'
output_dir = 'runs/detect'

csv_YoloV11m = pd.read_csv(csv_YoloV11m)
csv_YoloV11n = pd.read_csv(csv_YoloV11n)

# Plot data from the first CSV file
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.bar(csv_YoloV11m['Label'], csv_YoloV11m['Count'], color='b')
ax1.set_title('Anzahl bei YoloV11m')
ax1.set_xlabel('Label')
ax1.set_ylabel('Anzahl')
ax1.set_xticklabels(csv_YoloV11m['Label'], rotation=90)

ax2.bar(csv_YoloV11m['Label'], csv_YoloV11m['Average Confidence'], color='g')
ax2.set_title('Average Confidence bei YoloV11m')
ax2.set_xlabel('Label')
ax2.set_ylabel('Average Confidence')
ax2.set_xticklabels(csv_YoloV11m['Label'], rotation=90)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(os.path.join(output_dir, 'YoloV11m.png'))
plt.close()

# Plot data from the second CSV file
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))

ax3.bar(csv_YoloV11n['Label'], csv_YoloV11n['Count'], color='b')
ax3.set_title('Anzahl bei YoloV11n')
ax3.set_xlabel('Label')
ax3.set_ylabel('Anzahl')
ax3.set_xticklabels(csv_YoloV11n['Label'], rotation=90)

ax4.bar(csv_YoloV11n['Label'], csv_YoloV11n['Average Confidence'], color='g')
ax4.set_title('Average Confidence bei YoloV11n')
ax4.set_xlabel('Label')
ax4.set_ylabel('Average Confidence')
ax4.set_xticklabels(csv_YoloV11n['Label'], rotation=90)

plt.savefig(os.path.join(output_dir, 'YoloV11n.png'))
plt.close()

# Calculate differences and plot
merged_data = pd.merge(csv_YoloV11m, csv_YoloV11n, on='Label', suffixes=('_1', '_2'))
merged_data['Count Difference'] = merged_data['Count_1'] - merged_data['Count_2']
merged_data['Average Confidence Difference'] = merged_data['Average Confidence_1'] - merged_data['Average Confidence_2']

fig, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 6))

ax5.bar(merged_data['Label'], merged_data['Count Difference'], color='b')
ax5.set_title('Anzahl Differenz')
ax5.set_xlabel('Label')
ax5.set_ylabel('Anzahl Differenz')
ax5.set_xticklabels(merged_data['Label'], rotation=90)

ax6.bar(merged_data['Label'], merged_data['Average Confidence Difference'], color='g')
ax6.set_title('Average Confidence Differenz')
ax6.set_xlabel('Label')
ax6.set_ylabel('Average Confidence Differenz')
ax6.set_xticklabels(merged_data['Label'], rotation=90)

plt.savefig(os.path.join(output_dir, 'difference_plot.png'))
plt.close()