import os
import numpy as np
import joblib
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

IMG_SIZE = 128
DATA_DIR = "dataset"

def load_and_preprocess(data_dir, img_size):
    X = []
    y = []
    for label, class_name in enumerate(["clean", "green"]):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_size, img_size))
                img_array = np.array(img).flatten()
                X.append(img_array)
                y.append(class_name)
            except Exception as e:
                print(f"Ошибка при загрузке {img_path}: {e}")
    return np.array(X), y

def learn():
    label_encoder = LabelEncoder()
    X, y = load_and_preprocess(DATA_DIR, IMG_SIZE)

    X = X / 255.0

    print(X)
    print(y)

    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=["clean", "green"]))

    joblib.dump(model, 'water_classifier.pkl')
    joblib.dump(label_encoder, 'water_labels.pkl')