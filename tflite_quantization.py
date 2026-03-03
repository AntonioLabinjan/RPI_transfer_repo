import os
import sys
import time
import numpy as np
import tensorflow as tf
import tf_keras as keras
import matplotlib.pyplot as plt
import akida
from pathlib import Path
from math import exp

# --- 1. POSTAVKE I PUTANJE ---
KERAS_MODEL_PATH = "./outputs/best_person_detection.keras"
TFLITE_FLOAT_PATH = "person_detection.tflite"
TFLITE_QUANT_PATH = "person_detection_quant.tflite"
AKIDA_MODEL_PATH = "./outputs/person_detection_pro.fbz"

TEST_IMAGES_DIR = Path("./yolo_dataset/images/test")
THRESHOLD = 0.5

# NumPy patchevi za stabilnost Akida importa
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'bool'): np.bool = bool

# --- 2. POMOĆNE FUNKCIJE ---

def sigmoid(x):
    return 1 / (1 + exp(-x)) if x >= 0 else exp(x) / (1 + exp(x))

def run_tflite_inference(model_path, img_input, is_quantized=False):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    if is_quantized:
        # TFLite Quant očekuje uint8 [0, 255]
        input_data = (img_input * 255).astype(np.uint8)
    else:
        input_data = img_input.astype(np.float32)
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if is_quantized:
        return output_data[0][0] / 255.0
    return output_data[0][0]

def get_img_for_model(img_path, size):
    image = tf.io.read_file(str(img_path))
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, [size, size])
    return tf.cast(image, tf.float32) / 255.0

# --- 3. GLAVNA USPOREDBA ---

def run_complete_comparison():
    print("🚀 Učitavam modele...")
    k_model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)
    a_model = akida.Model(AKIDA_MODEL_PATH)
    
    test_img_paths = sorted([p for p in TEST_IMAGES_DIR.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])[:5]
    
    if not test_img_paths:
        print("❌ Nema slika u test direktoriju!")
        return

    plt.figure(figsize=(20, 10))

    for i, img_path in enumerate(test_img_paths):
        # 1. Keras & TFLite (320x320)
        img_320 = get_img_for_model(img_path, 320).numpy()
        img_320_batch = np.expand_dims(img_320, axis=0)
        
        # 2. Akida (224x224)
        img_224 = get_img_for_model(img_path, 224).numpy()
        img_224_batch = (np.expand_dims(img_224, axis=0) * 255).astype(np.uint8)

        # INFERENCE POZIVI
        res_keras = k_model.predict(img_320_batch, verbose=0)[0][0]
        res_tflite_f = run_tflite_inference(TFLITE_FLOAT_PATH, img_320_batch, False)
        res_tflite_q = run_tflite_inference(TFLITE_QUANT_PATH, img_320_batch, True)
        
        akida_logit = a_model.predict(img_224_batch)[0][0]
        res_akida = sigmoid(akida_logit)

        def fmt(val): return "HUMAN" if val < THRESHOLD else "POZADINA"

        # Vizualizacija
        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(img_320.squeeze(), cmap='gray')
        title = (f"Keras: {fmt(res_keras)}\n"
                 f"TF-F: {fmt(res_tflite_f)}\n"
                 f"TF-Q: {fmt(res_tflite_q)}\n"
                 f"Akida: {fmt(res_akida)}")
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# --- 4. BENCHMARK (STATISTIKA) ---

def benchmark_all():
    print("\n" + "="*70)
    print(f"{'MODEL':<25} | {'SIZE (MB)':<12} | {'TIME (ms)':<12}")
    print("-" * 70)

    # Sample za benchmark
    sample_path = next(TEST_IMAGES_DIR.glob("*"))
    img_320 = np.expand_dims(get_img_for_model(sample_path, 320).numpy(), 0)
    img_224 = (np.expand_dims(get_img_for_model(sample_path, 224).numpy(), 0) * 255).astype(np.uint8)

    models_info = [
        ("Keras Original", KERAS_MODEL_PATH, "keras"),
        ("TFLite Float32", TFLITE_FLOAT_PATH, "tflite_f"),
        ("TFLite INT8", TFLITE_QUANT_PATH, "tflite_q"),
        ("Akida SNN (.fbz)", AKIDA_MODEL_PATH, "akida")
    ]

    # Ponovno učitavanje modela za čisti benchmark
    k_model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)
    a_model = akida.Model(AKIDA_MODEL_PATH)

    for name, path, m_type in models_info:
        if not os.path.exists(path):
            print(f"{name:<25} | Nema datoteke")
            continue
            
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        # Warm-up i mjerenje (10 iteracija)
        times = []
        for _ in range(11):
            start = time.time()
            if m_type == "keras": k_model.predict(img_320, verbose=0)
            elif m_type == "tflite_f": run_tflite_inference(path, img_320, False)
            elif m_type == "tflite_q": run_tflite_inference(path, img_320, True)
            elif m_type == "akida": a_model.predict(img_224)
            if _ > 0: times.append((time.time() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"{name:<25} | {size_mb:<12.2f} | {avg_time:<12.2f}")

if __name__ == "__main__":
    run_complete_comparison()
    benchmark_all()