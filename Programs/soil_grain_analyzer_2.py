import cv2
import numpy as np
import os
import csv
from skimage import measure

image_dir = r'C:\SPROS\Grain Size Analysis\Dataset\Sedimentary\Sandstone'
csv_file = 'grain_data_sandstone.csv'
fields = ['filename', 'mean_grain_diameter_px', 'grain_count']

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Basic thresholding (adjust for your images)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    # Find contours (grains)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diameters = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:  # Ignore tiny grains/noise
            continue
        # Compute equivalent diameter
        diameter = np.sqrt(4*area/np.pi)
        diameters.append(diameter)
    mean_diam = np.mean(diameters) if diameters else 0
    grain_count = len(diameters)
    return mean_diam, grain_count

dataset = []
for image_name in os.listdir(image_dir):
    if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    full_path = os.path.join(image_dir, image_name)
    mean_diam, grain_count = process_image(full_path)
    dataset.append({
        'filename': image_name,
        'mean_grain_diameter_px': round(mean_diam, 2),
        'grain_count': grain_count
    })

with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in dataset:
        writer.writerow(row)

print(f"Grain size analysis saved to {csv_file}")
