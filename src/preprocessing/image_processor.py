import cv2
import numpy as np
import os
from PIL import Image


def remove_background(image, lower_thresh, upper_thresh):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    return cv2.bitwise_and(image, image, mask=mask)


def preprocess_data(image_path, target_size=(224, 224), lower_thresh=None, upper_thresh=None, save_path=None):
    try:
        with Image.open(image_path) as img:
            image_cv = np.array(img)

            if lower_thresh is not None and upper_thresh is not None:
                image_cv = remove_background(image_cv, lower_thresh, upper_thresh)

            img = cv2.resize(image_cv, target_size, interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = np.array(img_rgb, dtype=np.float32) / 255.0

            if save_path is not None:
                filename = os.path.basename(image_path)
                save_filename = os.path.join(save_path, filename)
                cv2.imwrite(save_filename, img_rgb)
                print(f"Processed image saved to: {save_filename}")

            return img_array.flatten()

    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")