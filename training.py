# dodat negatiives u dataset => snimit s evk live feedon frames koji u sebi nemaju ljude
import os
import sys
import re
from pathlib import Path
from collections import Counter
import numpy as np

# --- 1. NAPREDNI PATCHEVI ---
if not hasattr(np, 'strings'):
    class DummyStrings:
        def encode(self, array, encoding='utf-8'):
            if np.isscalar(array) or getattr(array, 'ndim', 0) == 0:
                return np.array(str(array).encode(encoding))
            return np.array([str(s).encode(encoding) for s in array])
    np.strings = DummyStrings()

if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'bool'): np.bool = bool

# --- 2. IMPORTI ---
import tensorflow as tf
import tf_keras as keras
import akida
import matplotlib.pyplot as plt
from akida_models import akidanet_imagenet
from sklearn.metrics import accuracy_score, f1_score
from tf_keras import Input, Model, layers

# --- 3. KONFIGURACIJA ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 80  # Povećaj slobodno sad kad imaš više podataka

YOLO_DATASET_PATH = Path("./yolo_dataset")
OUTPUT_DIR = Path("./outputs")
BEST_SAVE_PATH = OUTPUT_DIR / "NEW_person_detection_w_negatives.keras"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 4. FUNKCIJE ZA PODATKE ---

def load_yolo_split(split: str):
    image_paths, labels = [], []
    img_dir = YOLO_DATASET_PATH / "images" / split
    lbl_dir = YOLO_DATASET_PATH / "labels" / split

    if not img_dir.exists(): 
        return np.array([]), np.array([], dtype=np.int32)

    valid_exts = (".png", ".jpg", ".jpeg")
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() in valid_exts:
            txt_path = lbl_dir / (img_path.stem + ".txt")
            # 0 = Human (ima labelu), 1 = Not Human (nema labelu ili je prazna)
            label = 0 if (txt_path.exists() and os.path.getsize(txt_path) > 0) else 1
            image_paths.append(str(img_path))
            labels.append(label)

    return np.array(image_paths), np.array(labels, dtype=np.int32)

def _group_key(path_str: str) -> str:
    name = Path(path_str).name
    base = name.split(".rf.")[0]
    m = re.search(r"autogen_frame_\d+_png", base)
    return m.group(0) if m else base

def de_leak_reassign(tr_img, tr_lbl, vl_img, vl_lbl, ts_img, ts_lbl):
    buckets = {}
    for s, imgs, lbls in [("train", tr_img, tr_lbl), ("valid", vl_img, vl_lbl), ("test", ts_img, ts_lbl)]:
        for p, y in zip(imgs, lbls):
            g = _group_key(str(p))
            buckets.setdefault(g, []).append((s, str(p), int(y)))

    out = {"train": [], "valid": [], "test": []}
    for _, items in buckets.items():
        present = {s for s, _, _ in items}
        target = "test" if "test" in present else ("valid" if "valid" in present else "train")
        out[target].extend((p, y) for _, p, y in items)

    def to_np(arr):
        if not arr: return np.array([]), np.array([], dtype=np.int32)
        p, y = zip(*arr)
        return np.array(p), np.array(y, dtype=np.int32)

    return (*to_np(out["train"]), *to_np(out["valid"]), *to_np(out["test"]))

def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image.set_shape([IMG_SIZE, IMG_SIZE, 1])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def train_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

# --- 5. MODEL ---

def create_akida_model():
    # Koristimo AkidaNet koji je optimiziran za neuromorfni hardver
    base_model = akidanet_imagenet(input_shape=(IMG_SIZE, IMG_SIZE, 1), include_top=False, pooling='avg')
    base_model.trainable = True 

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = base_model(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="logits")(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --- 6. IZVRŠAVANJE ---

tr_img, tr_lbl = load_yolo_split("train")
vl_img, vl_lbl = load_yolo_split("val")
ts_img, ts_lbl = load_yolo_split("test")

if len(tr_img) == 0:
    print("❌ Greška: Dataset je prazan!")
    sys.exit()

tr_img, tr_lbl, vl_img, vl_lbl, ts_img, ts_lbl = de_leak_reassign(tr_img, tr_lbl, vl_img, vl_lbl, ts_img, ts_lbl)

# Class Weights (bitno jer ćeš sad imati puno pozadine)
counts = Counter(tr_lbl)
total = len(tr_lbl)
class_weight = {
    0: (total / (2 * counts[0])) if counts[0] > 0 else 1.0, 
    1: (total / (2 * counts[1])) if counts[1] > 0 else 1.0  
}

train_ds = tf.data.Dataset.from_tensor_slices((tr_img, tr_lbl.astype(np.float32))) \
    .shuffle(len(tr_img)).map(parse_image).map(train_augment).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((vl_img, vl_lbl.astype(np.float32))) \
    .map(parse_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((ts_img, ts_lbl.astype(np.float32))) \
    .map(parse_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = create_akida_model()
callbacks = [
    keras.callbacks.ModelCheckpoint(str(BEST_SAVE_PATH), monitor="val_loss", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=1000, restore_best_weights=True)
]

print(f"\n🚀 Trening na {len(tr_img)} slika (Hum: {counts[0]}, No-Hum: {counts[1]})...")
model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, class_weight=class_weight, callbacks=callbacks)

# --- 7. FINALNA EVALUACIJA ---

model = keras.models.load_model(str(BEST_SAVE_PATH), compile=False)
v_probs = model.predict(valid_ds).flatten()
v_true = vl_lbl

# Optimizacija praga
best_thr, max_f1 = 0.5, 0
for t in np.linspace(0.1, 0.9, 81):
    f1 = f1_score(v_true, (v_probs >= t).astype(int), zero_division=0)
    if f1 > max_f1:
        max_f1, best_thr = f1, t

print(f"\n✅ Trening završen!")
print(f"🎯 Najbolji prag za Live Feed: {best_thr:.2f}")

def visual_check(model, ds, threshold):
    for imgs, lbls in ds.take(1):
        preds = model.predict(imgs)
        plt.figure(figsize=(12, 6))
        for i in range(min(4, len(imgs))):
            ax = plt.subplot(1, 4, i + 1)
            p = 1 if preds[i][0] >= threshold else 0
            color = 'green' if p == int(lbls[i]) else 'red'
            ax.imshow(imgs[i].numpy().squeeze(), cmap='gray')
            ax.set_title(f"Pred: {'Bg' if p==1 else 'Human'}\nReal: {'Bg' if int(lbls[i])==1 else 'Human'}", color=color)
            ax.axis('off')
        plt.savefig(OUTPUT_DIR / "final_check.png")
        print(f"📊 Vizualna provjera spremljena u {OUTPUT_DIR / 'final_check.png'}")

visual_check(model, test_ds, best_thr)