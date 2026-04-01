import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.image_processor import preprocess_data


# PATH 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#Lutein   
#Beta-carotene
#Lycopene
TARGET_NAME = "Lycopene"  # เปลี่ยนตรงนี้

print(f"Running preprocessing for: {TARGET_NAME}")


csv_file_path = os.path.join(BASE_DIR, "data", f"{TARGET_NAME}.csv")
base_image_path = os.path.join(BASE_DIR, "data", "raw")
processed_data_dir = os.path.join(BASE_DIR, "data", "processed", TARGET_NAME)

# Threshold values for background removal
lower_thresh = np.array([0, 70, 0])
upper_thresh = np.array([255, 255, 255])

target_size = (224, 224)

# ======================
# SETUP
# ======================
os.makedirs(processed_data_dir, exist_ok=True)

carotenoid_data = pd.read_csv(csv_file_path, header=None, names=['folder', 'carotenoid_level'])


# PROCESS DATA

image_data = []


def build_dataset(carotenoid_data, base_image_path, processed_data_dir):
    image_data = []

    for _, row in carotenoid_data.iterrows():
        folder, carotenoid_level = row['folder'], row['carotenoid_level']
        folder_path = os.path.join(base_image_path, folder)
        save_path = os.path.join(processed_data_dir, folder)

        os.makedirs(save_path, exist_ok=True)

        for i in range(1, 5):
            image_path = os.path.join(folder_path, f"{i}.jpg")

            if os.path.isfile(image_path):
                try:
                    img_data = preprocess_data(
                        image_path,
                        target_size,
                        lower_thresh,
                        upper_thresh,
                        save_path=save_path
                    )
                    image_data.append((img_data, carotenoid_level))
                except Exception as e:
                    print(e)

    return image_data


image_data = build_dataset(
    carotenoid_data,
    base_image_path,
    processed_data_dir
)

# ======================
# CONVERT TO ARRAY
# ======================
X = np.array([data for data, _ in image_data])
y = np.array([level for _, level in image_data])

# ======================
# SPLIT DATA
# ======================
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

# ======================
# LOG
# ======================
print("Training set size:", X_train_temp.shape)
print("Validation set size:", X_val.shape)
print("Test set size:", X_test.shape)

# ======================
# SAVE
# ======================
np.save(os.path.join(processed_data_dir, 'X_train_temp.npy'), X_train_temp)
np.save(os.path.join(processed_data_dir, 'y_train_temp.npy'), y_train_temp)
np.save(os.path.join(processed_data_dir, 'X_val.npy'), X_val)
np.save(os.path.join(processed_data_dir, 'y_val.npy'), y_val)
np.save(os.path.join(processed_data_dir, 'X_test.npy'), X_test)
np.save(os.path.join(processed_data_dir, 'y_test.npy'), y_test)

print("Processed data saved successfully.")