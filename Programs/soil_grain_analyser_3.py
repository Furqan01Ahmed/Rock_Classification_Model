import cv2
import numpy as np
import os
import csv
from sklearn.cluster import KMeans
import webcolors
from skimage import measure

image_dir = r'C:\SPROS\Grain Size Analysis\Dataset\Sedimentary\Sandstone'  # Update this path as needed
csv_file = 'grain_data_with_colors_sandstone.csv'   #Update name as needed
fields = ['filename', 'mean_grain_diameter_px', 'grain_count', 'dominant_colors', 'color_percentages']

def get_closest_color_name(rgb_color):
    """Find closest named color - compatible with different webcolors versions"""
    min_colors = {}
    
    # Try different attribute names for compatibility
    try:
        # For newer versions
        color_dict = webcolors.CSS3_HEX_TO_NAMES
    except AttributeError:
        try:
            # For older versions
            color_dict = webcolors.css3_hex_to_names
        except AttributeError:
            # Fallback - use a basic color mapping
            basic_colors = {
                '#000000': 'black', '#FFFFFF': 'white', '#FF0000': 'red',
                '#00FF00': 'green', '#0000FF': 'blue', '#FFFF00': 'yellow',
                '#FF00FF': 'magenta', '#00FFFF': 'cyan', '#808080': 'gray',
                '#800000': 'maroon', '#008000': 'darkgreen', '#000080': 'navy',
                '#A52A2A': 'brown', '#FFA500': 'orange', '#800080': 'purple'
            }
            color_dict = basic_colors
    
    for hex_code, name in color_dict.items():
        try:
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
        except:
            # Skip invalid hex codes
            continue
        
        rd = (r_c - rgb_color[0]) ** 2
        gd = (g_c - rgb_color[1]) ** 2
        bd = (b_c - rgb_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    
    return min_colors[min(min_colors.keys())] if min_colors else 'unknown'

def extract_dominant_colors(image_path, k=5):
    """Extract dominant colors from image"""
    img = cv2.imread(image_path)
    if img is None:
        return []
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (150, 150))  # Resize for faster processing
    img_data = img_resized.reshape((-1, 3))
    
    # K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(img_data)
    
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    
    # Calculate color percentages
    unique, counts = np.unique(labels, return_counts=True)
    percentages = (counts / len(labels)) * 100
    
    color_info = []
    for i, color in enumerate(colors):
        try:
            # Try exact match first
            closest_name = webcolors.rgb_to_name((color[0], color[1], color[2]))
        except (ValueError, AttributeError):
            # If exact match not found, find closest
            closest_name = get_closest_color_name((color[0], color[1], color[2]))
        
        color_info.append({
            'name': closest_name,
            'rgb': color,
            'percentage': round(percentages[i], 1)
        })
    
    # Sort by percentage (most dominant first)
    color_info.sort(key=lambda x: x['percentage'], reverse=True)
    
    return color_info

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, 0, 'unknown', 'unknown'
        
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
    
    # Extract color information
    color_info = extract_dominant_colors(image_path)
    if color_info:
        dominant_colors = ', '.join([c['name'] for c in color_info[:3]])  # Top 3 colors
        color_percentages = ', '.join([f"{c['name']}: {c['percentage']}%" for c in color_info[:3]])
    else:
        dominant_colors = 'unknown'
        color_percentages = 'unknown'
    
    return mean_diam, grain_count, dominant_colors, color_percentages

dataset = []
for image_name in os.listdir(image_dir):
    if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    full_path = os.path.join(image_dir, image_name)
    try:
        mean_diam, grain_count, colors, percentages = process_image(full_path)
        dataset.append({
            'filename': image_name,
            'mean_grain_diameter_px': round(mean_diam, 2),
            'grain_count': grain_count,
            'dominant_colors': colors,
            'color_percentages': percentages
        })
        print(f"Processed: {image_name}")
    except Exception as e:
        print(f"Error processing {image_name}: {e}")

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in dataset:
        writer.writerow(row)

print(f"Grain size and color analysis saved to {csv_file}")
