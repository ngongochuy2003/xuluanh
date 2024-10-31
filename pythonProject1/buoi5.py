import cv2
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Hàm xử lý và chuyển ảnh sang xám từ thư mục
def load_and_preprocess_images(image_folder):
    data = []
    labels = os.listdir(image_folder)

    for label in labels:
        label_folder = os.path.join(image_folder, label)
        if not os.path.isdir(label_folder):
            continue

        for filename in os.listdir(label_folder):
            image_path = os.path.join(label_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Chuyển ảnh sang xám và resize
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (64, 64))
            flat_image = resized_image.flatten()

            data.append(np.append(flat_image, label))

    return np.array(data)


# Cây quyết định sử dụng thuật toán CART (Gini) và ID3 (Entropy)
def gini_index(groups, classes):
    n_instances = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini


def entropy(groups, classes):
    n_instances = sum([len(group) for group in groups])
    entropy = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            if p != 0:
                score += p * np.log2(p)
        entropy += (-score) * (size / n_instances)
    return entropy


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def get_split_gini(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def get_split_entropy(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, -999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
groups = test_split(index, row[index], dataset)
ent = entropy(groups, class_values)
if ent > b_score:
    b_index, b_value, b_score, b_groups = index, row[index], ent, groups
return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth, criterion):
    left, right = node['groups']
    del (node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split_gini(left) if criterion == 'gini' else get_split_entropy(left)
        split(node['left'], max_depth, min_size, depth + 1, criterion)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split_gini(right) if criterion == 'gini' else get_split_entropy(right)
        split(node['right'], max_depth, min_size, depth + 1, criterion)


def build_tree(train, max_depth, min_size, criterion='gini'):
    root = get_split_gini(train) if criterion == 'gini' else get_split_entropy(train)
    split(root, max_depth, min_size, 1, criterion)
    return root


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Xử lý dữ liệu IRIS
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
iris_dataset = np.column_stack((X_iris, y_iris))

# Chia tập huấn luyện và kiểm tra cho IRIS
train_data_iris, test_data_iris = train_test_split(iris_dataset, test_size=0.3, random_state=42)

# Xây dựng cây quyết định cho IRIS
tree_gini_iris = build_tree(train_data_iris, max_depth=3, min_size=1, criterion='gini')
tree_entropy_iris = build_tree(train_data_iris, max_depth=3, min_size=1, criterion='entropy')

# Dự đoán và đánh giá mô hình IRIS
y_test_iris = [row[-1] for row in test_data_iris]
y_pred_gini_iris = [predict(tree_gini_iris, row) for row in test_data_iris]
y_pred_entropy_iris = [predict(tree_entropy_iris, row) for row in test_data_iris]

print("IRIS - CART (Gini) Accuracy:", accuracy_score(y_test_iris, y_pred_gini_iris))
print("IRIS - CART (Gini) Classification Report:\n", classification_report(y_test_iris, y_pred_gini_iris))

print("IRIS - ID3 (Entropy) Accuracy:", accuracy_score(y_test_iris, y_pred_entropy_iris))
print("IRIS - ID3 (Entropy) Classification Report:\n", classification_report(y_test_iris, y_pred_entropy_iris))
# Xử lý dữ liệu từ thư mục ảnh nha khoa
image_folder = "path/to/your/image/folder"
dataset_images = load_and_preprocess_images(image_folder)

# Chia tập huấn luyện và kiểm tra cho ảnh nha khoa
train_data_images, test_data_images = train_test_split(dataset_images, test_size=0.3, random_state=42)

# Xây dựng cây quyết định cho ảnh nha khoa
tree_gini_images = build_tree(train_data_images, max_depth=3, min_size=1, criterion='gini')
tree_entropy_images = build_tree(train_data_images, max_depth=3, min_size=1, criterion='entropy')

# Dự đoán và đánh giá mô hình ảnh nha khoa
y_test_images = [row[-1] for row in test_data_images]
y_pred_gini_images = [predict(tree_gini_images, row) for row in test_data_images]
y_pred_entropy_images = [predict(tree_entropy_images, row) for row in test_data_images]

print("Dental Images - CART (Gini) Accuracy:", accuracy_score(y_test_images, y_pred_gini_images))
print("Dental Images - CART (Gini) Classification Report:\n", classification_report(y_test_images, y_pred_gini_images))

print("Dental Images - ID3 (Entropy) Accuracy:", accuracy_score(y_test_images, y_pred_entropy_images))
print("Dental Images - ID3 (Entropy) Classification Report:\n",
      classification_report(y_test_images, y_pred_entropy_images))