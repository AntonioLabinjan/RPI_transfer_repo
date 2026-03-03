import os
import numpy as np
import tensorflow as tf
import tf_keras as keras
import matplotlib.pyplot as plt
import akida
from pathlib import Path
from math import exp

# --- 1. KONFIGURACIJA I PATCHEVI ---
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'bool'): np.bool = bool

# Putanje
OUTPUT_DIR = Path("./outputs")
KERAS_MODEL_PATH = OUTPUT_DIR / "best_person_detection.keras"
TFLITE_MODEL_PATH = OUTPUT_DIR / "person_detection.tflite"
AKIDA_MODEL_PATH = OUTPUT_DIR / "person_detection_pro.fbz"
TEST_IMAGES_DIR = Path("./yolo_dataset/images/test")

# Parametri (Modeli mogu imati razlicite ulaze)
KERAS_SIZE = 320
AKIDA_SIZE = 224
THRESHOLD = 0.5  # Standardni prag, prilagodi po potrebi

# --- 2. POMOCNE FUNKCIJE ---

def sigmoid(x):
    """Pretvara Akida logit u vjerojatnost [0, 1]"""
    try:
        return 1 / (1 + exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def prepare_input(filename, size):
    """Ucitava i resiza sliku za specificni model"""
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, [size, size])
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)

def run_tflite_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# --- 3. GLAVNI INFERENCE SCRIPT ---

def run_comparison():
    # 1. Ucitavanje svih modela
    print("🔄 Ucitavam modele...")
    keras_model = keras.models.load_model(str(KERAS_MODEL_PATH), compile=False)
    
    interpreter = None
    if TFLITE_MODEL_PATH.exists():
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
        interpreter.allocate_tensors()
        print("✅ TFLite ucitan.")

    akida_model = None
    if AKIDA_MODEL_PATH.exists():
        akida_model = akida.Model(str(AKIDA_MODEL_PATH))
        print("✅ Akida FBZ ucitan.")

    # 2. Priprema testnih slika
    test_img_paths = [str(p) for p in TEST_IMAGES_DIR.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    if not test_img_paths:
        print("❌ Nema slika u test direktoriju!")
        return

    # 3. Vizualizacija (2 reda po 5 slika)
    plt.figure(figsize=(22, 12))
    num_samples = min(10, len(test_img_paths))
    
    for i in range(num_samples):
        img_path = test_img_paths[i]
        
        # --- INFERENCE ---
        # Keras (320x320)
        keras_in = prepare_input(img_path, KERAS_SIZE)
        keras_prob = keras_model.predict(keras_in, verbose=0)[0][0]
        
        # TFLite (320x320 - pretpostavka da je isti kao Keras)
        tflite_prob = run_tflite_inference(interpreter, keras_in.numpy()) if interpreter else -1
        
        # Akida (224x224)
        akida_prob = -1
        if akida_model:
            # Akida zahtijeva uint8 ulaz (0-255) za predikciju ako je model kvantiziran
            akida_in = prepare_input(img_path, AKIDA_SIZE).numpy() * 255
            akida_in = akida_in.astype('uint8')
            logit = akida_model.predict(akida_in)[0][0]
            akida_prob = sigmoid(logit)

        # --- LABELIRANJE ---
        # 0 = HUMAN, 1 = POZADINA (prema tvom loaderu)
        def get_label(p):
            if p == -1: return "N/A"
            return "HUMAN" if p < THRESHOLD else "POZADINA"

        # Prikaz slike (koristimo 320x320 verziju za prikaz)
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(keras_in.numpy().squeeze(), cmap='gray')
        
        # Generiranje naslova
        title = f"Keras: {get_label(keras_prob)}"
        if interpreter:
            title += f"\nTFLite: {get_label(tflite_prob)}"
        if akida_model:
            title += f"\nAkida: {get_label(akida_prob)}"
            
        ax.set_title(title, fontsize=10, pad=10)
        ax.axis('off')

    plt.tight_layout()
    save_path = OUTPUT_DIR / "triple_inference_comparison.png"
    plt.savefig(save_path)
    plt.show()
    print(f"\n📊 Usporedni test zavrsen! Rezultat spremljen: {save_path}")

if __name__ == "__main__":
    run_comparison()