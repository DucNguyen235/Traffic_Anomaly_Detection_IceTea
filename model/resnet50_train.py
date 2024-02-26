import os
import numpy as np
from sklearn.svm import OneClassSVM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib
import tensorflow as tf

# Define the path to your "normal" class dataset folder
normal_class_dir = '/content/drive/MyDrive/pattern_data/normal_clean'  # Change to the path of your "normal" class folder

# Load a pre-trained ResNet50 model as a feature extractor
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Extract features from "normal" class images
def extract_features_from_images(image_paths, model):
    features = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        features.append(feature)
    return np.array(features)

# Collect image file paths for the "normal" class
normal_image_paths = [os.path.join(normal_class_dir, img_name) for img_name in os.listdir(normal_class_dir)]

# Extract features from "normal" class images
normal_features = extract_features_from_images(normal_image_paths, base_model)

# Flatten the feature vectors for the "normal" class
normal_features_flat = normal_features.reshape(normal_features.shape[0], -1)

# Train a One-Class SVM model with RBF kernel and gamma='scale'
one_class_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.000003)  # Change the kernel and gamma
one_class_svm.fit(normal_features_flat)

# Define the path where you want to save the model
model_save_path = '/content/drive/MyDrive/model/one_class_svm_rbf_0.00001_vn_recal96.pkl'  # Change the path as needed

# Save the One-Class SVM model to a file
joblib.dump(one_class_svm, model_save_path)

print(f"Model saved to {model_save_path}")