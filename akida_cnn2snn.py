import os
import sys
import re
from pathlib import Path
from collections import Counter
import numpy as np

# --- 0. OBAVEZNI NUMPY PATCHEVI ---
if not hasattr(np, 'strings'):
    class DummyStrings:
        def encode(self, array, encoding='utf-8'):
            if np.isscalar(array) or getattr(array, 'ndim', 0) == 0:
                return np.array(str(array).encode(encoding))
            return np.array([str(s).encode(encoding) for s in array])
    np.strings = DummyStrings()

if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'bool'): np.bool = bool

# --- 1. IMPORTI ---
import tensorflow as tf
import tf_keras as keras
import cnn2snn
from tf_keras import Input, Model, layers

# --- 2. KONFIGURACIJA ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = 224 
BATCH_SIZE = 16
EPOCHS = 1

YOLO_DATASET_PATH = Path("./yolo_dataset")
OUTPUT_DIR = Path("./outputs")
AKIDA_MODEL_PATH = OUTPUT_DIR / "person_detection_pro.fbz"
BEST_KERAS_PATH = OUTPUT_DIR / "person_detection_pro.keras"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 3. DATA LOADING ---
def load_yolo_split(split: str):
    image_paths, labels = [], []
    img_dir = YOLO_DATASET_PATH / "images" / split
    lbl_dir = YOLO_DATASET_PATH / "labels" / split
    
    if not img_dir.exists(): 
        print(f"⚠️ Upozorenje: Direktorij {img_dir} ne postoji.")
        return np.array([]), np.array([], dtype=np.int32)
    
    valid_exts = (".png", ".jpg", ".jpeg")
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() in valid_exts:
            txt_path = lbl_dir / (img_path.stem + ".txt")
            # 0 = Čovjek (ako postoji label file i nije prazan), 1 = Pozadina (ako ne postoji ili je prazan)
            is_human = 0 if (txt_path.exists() and os.path.getsize(txt_path) > 0) else 1
            image_paths.append(str(img_path))
            labels.append(is_human)
            
    return np.array(image_paths), np.array(labels, dtype=np.int32)

def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def train_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    return image, label

# --- 4. ARHITEKTURA ---
def create_pro_akida_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    def conv_block(x, filters, strides=1):
        x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    x = conv_block(inputs, 32, strides=2)
    x = conv_block(x, 64, strides=2)
    x = conv_block(x, 64, strides=1)       
    x = conv_block(x, 128, strides=2)
    x = conv_block(x, 128, strides=1)
    x = conv_block(x, 256, strides=2)
    
    x = layers.Conv2D(512, (3, 3), strides=1, padding='same', use_bias=False)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.ReLU()(x)
    
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation=None, name='logits')(x)
    
    base_model = Model(inputs, outputs, name="akida_pro_base")
    model = cnn2snn.quantize(base_model, weight_quantization=4, activ_quantization=4)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model

# --- 5. IZVRŠAVANJE ---
tr_img, tr_lbl = load_yolo_split("train")
vl_img, vl_lbl = load_yolo_split("val")

if len(tr_img) == 0:
    print("❌ Greška: Dataset 'train' je prazan. Provjeri putanje!")
    sys.exit()

# Siguran izračun Class Weights
counts = Counter(tr_lbl)
total = len(tr_lbl)
print(f"📊 Statistika trening seta: Čovjek: {counts[0]}, Pozadina: {counts[1]}")

cw = {}
for cls in [0, 1]:
    if counts[cls] > 0:
        cw[cls] = total / (2.0 * counts[cls])
    else:
        cw[cls] = 1.0  # Default ako nema klase (da ne pukne)

train_ds = tf.data.Dataset.from_tensor_slices((tr_img, tr_lbl.astype(np.float32))) \
    .shuffle(len(tr_img)).map(parse_image).map(train_augment).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((vl_img, vl_lbl.astype(np.float32))) \
    .map(parse_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = create_pro_akida_model()

callbacks = [
    keras.callbacks.ModelCheckpoint(str(BEST_KERAS_PATH), save_best_only=True, monitor='val_loss')
]

print(f"\n🚀 Pokrećem PRO trening...")
model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, class_weight=cw, callbacks=callbacks)

# --- 6. KONVERZIJA ---
print("\n🔧 Konvertiram u Akida FBZ...")
try:
    akida_model = cnn2snn.convert(model)
    akida_model.save(str(AKIDA_MODEL_PATH))
    print(f"✅✅✅ USPJEH! Model spremljen u: {AKIDA_MODEL_PATH}")
except Exception as e:
    print(f"❌ Konverzija nije uspjela: {e}")