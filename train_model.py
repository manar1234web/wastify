import os, time, hashlib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import joblib

BASE_DIR = r"C:\Users\Manar\OneDrive\سطح المكتب\Material-Memory"

def find_data_root(root):
    for r, dirs, _ in os.walk(root):
        if "TRAIN" in dirs and "TEST" in dirs:
            return r
    return root

DATA_PATH = find_data_root(BASE_DIR)
print("مسار الداتا:", DATA_PATH)

rows = []
for split in ["TRAIN", "TEST"]:
    for label in ["O", "R"]:
        folder = os.path.join(DATA_PATH, split, label)
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                rows.append({"path": os.path.join(folder, name), "split": split, "label": label})

df = pd.DataFrame(rows)
print("إجمالي الصور:", len(df))

def check_image(path):
    try:
        with Image.open(path) as im:
            im.load()
            return im.size[0], im.size[1], True
    except Exception:
        return None, None, False

checked = [check_image(p) for p in df["path"]]
df[["width", "height", "ok"]] = pd.DataFrame(checked, index=df.index)
df = df[df["ok"]]
df = df[df[["width", "height"]].min(axis=1) >= 32]

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

df["hash"] = [file_hash(p) for p in df["path"]]
labels_per_hash = df.groupby("hash")["label"].nunique()
df = df[~df["hash"].isin(set(labels_per_hash[labels_per_hash > 1].index))]
train_hashes = set(df.loc[df["split"] == "TRAIN", "hash"])
df = df[~((df["split"] == "TEST") & df["hash"].isin(train_hashes))]
df = df.drop_duplicates(subset=["split", "hash"], keep="first").reset_index(drop=True)
print("بعد التنظيف:", len(df))

df["subset"] = "test"
train_part = df[df["split"] == "TRAIN"]
tr_idx, val_idx = train_test_split(train_part.index, test_size=0.15, stratify=train_part["label"], random_state=42)
df.loc[tr_idx, "subset"] = "train"
df.loc[val_idx, "subset"] = "val"

def extract_features(path):
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((128, 128))
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
    gray = np.asarray(rgb.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
    h = np.histogram(hsv[..., 0], bins=18, range=(0, 256))[0]
    s = np.histogram(hsv[..., 1], bins=8, range=(0, 256))[0]
    v = np.histogram(hsv[..., 2], bins=8, range=(0, 256))[0]
    colour = np.concatenate([h, s, v]) / (128.0 * 128.0)
    shape = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return np.concatenate([colour, shape]).astype(np.float32)

X = np.array([extract_features(p) for p in df["path"]], dtype=np.float32)
y = (df["label"] == "R").astype(int).values

X_train, y_train = X[df["subset"] == "train"], y[df["subset"] == "train"]
X_val, y_val = X[df["subset"] == "val"], y[df["subset"] == "val"]

best_tree, best_tree_f1 = None, -1
for depth in [5, 10, 15, 20]:
    tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=42)
    tree.fit(X_train, y_train)
    f1 = f1_score(y_val, tree.predict(X_val), average="macro")
    if f1 > best_tree_f1:
        best_tree, best_tree_f1 = tree, f1

best_svm, best_svm_f1 = None, -1
for c_val in [1, 10]:
    t0 = time.time()
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c_val, class_weight="balanced"))
    svm.fit(X_train, y_train)
    f1 = f1_score(y_val, svm.predict(X_val), average="macro")
    print(f"SVM C={c_val} انتهى في {time.time() - t0:.1f} ثانية | val F1={f1:.3f}")
    if f1 > best_svm_f1:
        best_svm, best_svm_f1 = svm, f1

joblib.dump(best_tree, os.path.join(BASE_DIR, "tree_model.joblib"))
joblib.dump(best_svm, os.path.join(BASE_DIR, "svm_model.joblib"))
print("تم حفظ tree_model.joblib و svm_model.joblib بنجاح.")
