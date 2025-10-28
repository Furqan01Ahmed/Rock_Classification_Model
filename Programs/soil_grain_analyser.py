import cv2
import numpy as np
import os
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

import warnings
warnings.filterwarnings('ignore')

ROCK_DATABASE = {
    'sandstone': {
        'colors': ['tan', 'brown', 'red', 'white', 'yellow'],
        'minerals': ['quartz', 'feldspar', 'mica', 'clay minerals'],
        'composition': 'quartz (>50%), feldspar, rock fragments',
        'formation': 'cemented sand grains in beaches, deserts, rivers',
        'location': 'coastal areas, desert environments, river channels',
        'properties': 'medium-grained, porous, variable hardness'
    },
    'limestone': {
        'colors': ['white', 'gray', 'cream', 'light_brown'],
        'minerals': ['calcite', 'aragonite', 'dolomite'],
        'composition': 'calcium carbonate (>50%), fossils',
        'formation': 'marine organisms, chemical precipitation',
        'location': 'shallow marine environments, coral reefs',
        'properties': 'fine to coarse-grained, reactive to acid, soluble'
    },
    'coal': {
        'colors': ['black', 'dark_brown'],
        'minerals': ['carbon', 'pyrite', 'clay minerals'],
        'composition': 'organic carbon (>50%), plant remains',
        'formation': 'compressed plant material in swamps',
        'location': 'ancient swamps, peat bogs',
        'properties': 'lightweight, combustible, layered'
    }
}


class RockClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rock_classes = list(ROCK_DATABASE.keys())
        
    def extract_features(self, image_path):
        """Extract comprehensive features from rock image"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # 1. Color features
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean_rgb = np.mean(img_rgb.reshape(-1, 3), axis=0)
        std_rgb = np.std(img_rgb.reshape(-1, 3), axis=0)
        features.extend(mean_rgb)
        features.extend(std_rgb)
        
        # 2. Texture features (Local Binary Pattern)
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        features.extend(lbp_hist)
        
        # 3. Gray-Level Co-occurrence Matrix features
        glcm = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        energy = greycoprops(glcm, 'energy')[0, 0]
        features.extend([contrast, dissimilarity, homogeneity, energy])
        
        # 4. Statistical features
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = np.mean(((gray - mean_intensity) / std_intensity) ** 3)
        kurtosis = np.mean(((gray - mean_intensity) / std_intensity) ** 4)
        features.extend([mean_intensity, std_intensity, skewness, kurtosis])
        
        # 5. Edge features
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        return np.array(features)
    
    def train_model(self, image_dir):
        """Train the classifier on labeled images"""
        print("Training rock classifier...")
        
        X = []
        y = []
        
        for rock_type in self.rock_classes:
            rock_folder = os.path.join(image_dir, rock_type)
            if not os.path.exists(rock_folder):
                print(f"Warning: No folder found for {rock_type}")
                continue
                
            for image_name in os.listdir(rock_folder):
                if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                    
                image_path = os.path.join(rock_folder, image_name)
                try:
                    features = self.extract_features(image_path)
                    X.append(features)
                    y.append(rock_type)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        
        if len(X) == 0:
            print("No training data found!")
            return
            
        X = np.array(X)
        y = np.array(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Save model
        joblib.dump(self.model, 'rock_classifier.pkl')
        print("Model saved as rock_classifier.pkl")
    
    def load_model(self, model_path='rock_classifier.pkl'):
        """Load pre-trained model"""
        self.model = joblib.load(model_path)
        print("Model loaded successfully")
    
    def predict_rock_type(self, image_path):
        """Predict rock type from image"""
        try:
            features = self.extract_features(image_path)
            features = features.reshape(1, -1)
            
            # Get prediction and confidence
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            return prediction, confidence
        except Exception as e:
            print(f"Error predicting {image_path}: {e}")
            return 'unknown', 0.0

def analyze_grain_size(image_path):
    """Analyze grain size (from previous script)"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    diameters = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        diameter = np.sqrt(4*area/np.pi)
        diameters.append(diameter)
    
    mean_grain_size = np.mean(diameters) if diameters else 0
    grain_count = len(diameters)
    
    return mean_grain_size, grain_count

def process_dataset_with_ml(image_dir, output_csv='ml_rock_analysis.csv'):
    """Process entire dataset using ML classification"""
    
    # Initialize classifier
    classifier = RockClassifier()
    
    # Check if model exists, otherwise train
    if os.path.exists('rock_classifier.pkl'):
        classifier.load_model()
    else:
        print("No pre-trained model found. Please organize your images in folders by rock type and run training first.")
        return
    
    fields = ['filename', 'predicted_rock', 'confidence', 'dominant_colors', 'minerals', 
              'composition', 'formation_process', 'formation_location', 'properties', 
              'mean_grain_size_px', 'grain_count']
    
    dataset = []
    
    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        image_path = os.path.join(image_dir, image_name)
        
        # Predict rock type
        rock_type, confidence = classifier.predict_rock_type(image_path)
        
        # Get grain analysis
        mean_grain_size, grain_count = analyze_grain_size(image_path)
        
        # Get rock info from database
        rock_info = ROCK_DATABASE.get(rock_type, {
            'colors': ['unknown'],
            'minerals': 'unknown',
            'composition': 'unknown',
            'formation': 'unknown',
            'location': 'unknown',
            'properties': 'unknown'
        })
        
        analysis = {
            'filename': image_name,
            'predicted_rock': rock_type,
            'confidence': round(confidence, 3),
            'dominant_colors': ', '.join(rock_info['colors']),
            'minerals': ', '.join(rock_info['minerals']) if isinstance(rock_info['minerals'], list) else rock_info['minerals'],
            'composition': rock_info['composition'],
            'formation_process': rock_info['formation'],
            'formation_location': rock_info['location'],
            'properties': rock_info['properties'],
            'mean_grain_size_px': round(mean_grain_size, 2),
            'grain_count': grain_count
        }
        
        dataset.append(analysis)
        print(f"Processed: {image_name} -> {rock_type} ({confidence:.2f} confidence)")
    
    # Save results
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)
    
    print(f"Analysis complete! Results saved to {output_csv}")

# Usage Example:
if __name__ == "__main__":
    # For training (organize images in folders by rock type first):
    classifier = RockClassifier()
    classifier.train_model('training_images/')  # Folder with subfolders for each rock type
    
    # For processing your dataset:
    try:
        # ✅ Option 1: Use raw string
        process_dataset_with_ml(r"C:\SPROS\Grain Size Analysis\Dataset\Igneous\Granite", 'comprehensive_rock_analysis.csv')
        print("Dataset processing completed successfully 1.")
    except Exception as e:
        # ✅ Option 2: Escape backslashes
        process_dataset_with_ml("C:\\SPROS\\Grain Size Analysis\\Dataset\\Igneous\\Granite", 'comprehensive_rock_analysis.csv')
        print("Dataset processing completed successfully 2.")
    except Exception as e:
        # ✅ Option 3: Use forward slashes (Python accepts them on Windows)
        process_dataset_with_ml("C:/SPROS/Grain Size Analysis/Dataset/Igneous/Granite", 'comprehensive_rock_analysis.csv')
        print("Dataset processing completed successfully 3.")
    except Exception as e:
        print(f"Error processing dataset: {e}")
    