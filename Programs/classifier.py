import cv2
import numpy as np
import joblib
import pandas as pd
from sklearn.cluster import KMeans

# Load the trained model and label encoder
model = joblib.load(r'C:\SPROS\Grain Size Analysis\Programs\rock_csv_classifier.pkl')
le_type = joblib.load(r'C:\SPROS\Grain Size Analysis\Programs\rock_label_encoder.pkl')

def get_closest_color_name(rgb_color):
    basic_colors = {
        'black': (0,0,0), 'white':(255,255,255), 'brown':(165,42,42),
        'gray':(128,128,128), 'tan':(210,180,140), 'red':(255,0,0),
        'yellow':(255,255,0), 'green':(0,128,0)
    }
    distances = {name: np.sum((np.array(rgb_color) - np.array(rgb))**2) for name, rgb in basic_colors.items()}
    return min(distances, key=distances.get)

def extract_features_from_image(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image_path)
    if img_gray is None or img_color is None:
        raise ValueError(f"Error loading image: {image_path}")
    # Grain analysis
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diameters = [np.sqrt(4*cv2.contourArea(cnt)/np.pi) for cnt in contours if cv2.contourArea(cnt) >= 10]
    mean_diam = np.mean(diameters) if diameters else 0
    grain_count = len(diameters)
    # Color detection
    img_resized = cv2.resize(img_color, (100, 100))
    pixels = img_resized.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels)
    dominant_rgb = kmeans.cluster_centers_[0].astype(int)
    main_color = get_closest_color_name(tuple(dominant_rgb))
    # Encoding for classifier
    color_mapping = {'black':0, 'white':1, 'brown':2, 'gray':3, 'tan':4, 'red':5, 'yellow':6, 'green':7}
    main_color_enc = color_mapping.get(main_color, 0)
    # Also return color as string!
    return [mean_diam, grain_count, main_color_enc], mean_diam, grain_count, main_color

# Main
image_path = input("Enter path to rock image: ").strip()
try:
    features, mean_diam, grain_count, main_color = extract_features_from_image(image_path)
    features_df = pd.DataFrame([features], columns=['mean_grain_diameter_px', 'grain_count', 'main_color_encoded'])
    rock_label_pred = model.predict(features_df)[0]
    rock_type_name = le_type.inverse_transform([rock_label_pred])[0]
    print(f"\nResults for image: {image_path}")
    print(f"Predicted rock type: {rock_type_name} (label {rock_label_pred})")
    print(f"Mean grain size: {mean_diam:.2f} pixels")
    print(f"Grain count: {grain_count}")
    print(f"Dominant color: {main_color}")
except Exception as e:
    print("Error processing image:", e)
