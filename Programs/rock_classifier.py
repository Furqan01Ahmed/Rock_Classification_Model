import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


csv_folder = r'C:\SPROS\Grain Size Analysis\Processed_Data_03'  # <- Your folder path

#collect all CSV files
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
df_list = []

for file in csv_files:
    full_path = os.path.join(csv_folder, file)
    df = pd.read_csv(full_path)
    # Extract rock type from file name (everything before '.csv')
    rock_type = os.path.splitext(file)[0]
    df['rock_type'] = rock_type
    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(df_all)} records from {len(csv_files)} files.")

# Feature engineering
# Handle color feature if present
if 'dominant_colors' in df_all.columns:
    df_all['main_color'] = df_all['dominant_colors'].apply(lambda x: x.split(',')[0].strip())

features = ['mean_grain_diameter_px', 'grain_count']
le_color = None
if 'main_color' in df_all.columns:
    le_color = LabelEncoder()
    df_all['main_color_encoded'] = le_color.fit_transform(df_all['main_color'])
    features.append('main_color_encoded')

# Encode labels for the target
le_type = LabelEncoder()
df_all['rock_type_encoded'] = le_type.fit_transform(df_all['rock_type'])
target = 'rock_type_encoded'

X = df_all[features]
y = df_all[target]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Classification accuracy: {acc:.2f}")

# save model and encoders
joblib.dump(model, 'rock_csv_classifier.pkl')
joblib.dump(le_type, 'rock_label_encoder.pkl')
if le_color is not None:
    joblib.dump(le_color, 'rock_color_encoder.pkl')

print("Model saved as rock_csv_classifier.pkl")
print("Rock label encoder saved as rock_label_encoder.pkl")
if le_color is not None:
    print("Color label encoder saved as rock_color_encoder.pkl")
