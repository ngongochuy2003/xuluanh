import cv2
import numpy as np
import os
import time
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors, tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# Hàm tiền xử lý ảnh và trích xuất đặc trưng bằng HOG
def preprocess_image(image_path, img_size=(32, 32)):
    # Đọc ảnh và chuyển thành ảnh xám
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)

    # Trích xuất đặc trưng HOG
    features, hog_image = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )

    return features  # Trả về vector đặc trưng HOG


# Hàm tải ảnh và gán nhãn
def load_data(dataset_path):
    X, y = [], []
    class_names = os.listdir(dataset_path)
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img_features = preprocess_image(img_path)  # Trích xuất đặc trưng
            X.append(img_features)
            y.append(class_name)
    return np.array(X), np.array(y)


# Hàm huấn luyện và đánh giá mô hình SVM
def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    model = svm.SVC(kernel='linear')
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print("\nClassification Report for SVM:")
    print(classification_report(y_test, y_pred))

    return {
        "Model": "SVM",
        "Training Time (s)": end_time - start_time,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    }


# Hàm huấn luyện và đánh giá mô hình KNN
def train_and_evaluate_knn(X_train, y_train, X_test, y_test):
    model = neighbors.KNeighborsClassifier(n_neighbors=3)
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print("\nClassification Report for KNN:")
    print(classification_report(y_test, y_pred))

    return {
        "Model": "KNN",
        "Training Time (s)": end_time - start_time,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    }