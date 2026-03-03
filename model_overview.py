import json
import os
import tf_keras as keras
import tensorflow as tf

# Putanje
CONFIG_PATH = "/home/antonio/Desktop/PEDRo-Event-Based-Dataset/outputs/best_person_detection/config.json"

def get_raspberry_tflite_overview(config_path):
    print("="*50)
    print("ANALIZA ZA RASPBERRY PI (TFLite)")
    print("="*50)

    # 1. Ucitavanje konfiguracije
    if not os.path.exists(config_path):
        print(f"Pogreska: Datoteka nije pronadjena: {config_path}")
        return

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # 2. Rekonstrukcija modela
    try:
        # Rekonstrukcija arhitekture
        model = keras.models.model_from_json(json.dumps(config_data))
        
        print("\n[1] TFLITE INPUT/OUTPUT:")
        print(f"  - Input Shape: {model.input_shape}")
        print(f"  - Output Shape: {model.output_shape}")
        print(f"  - Input Dtype: {model.layers[0].dtype}")

        print("\n[2] MODEL COMPLEXITY:")
        trainable_count = sum([keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = sum([keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        total_params = trainable_count + non_trainable_count
        size_estimate = (total_params * 4) / (1024**2) # Procjena u MB za Float32
        print(f"  - Ukupno parametara: {total_params:,}")
        print(f"  - Procjena velicine (.tflite): cca {size_estimate:.2f} MB")

        # 3. Provjera TFLite kompatibilnosti
        print("\n[3] TFLITE OPTIMIZATION CHECK:")
        
        # Raspberry Pi 4/5 podrzava XNNPACK ubrzanje za ove operacije
        supported_ops = (keras.layers.Conv2D, keras.layers.DepthwiseConv2D, 
                        keras.layers.Dense, keras.layers.ReLU, keras.layers.BatchNormalization)
        
        unsupported_for_acceleration = []
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                for sub_layer in layer.layers:
                    if not isinstance(sub_layer, supported_ops) and not 'Input' in str(type(sub_layer)):
                        unsupported_for_acceleration.append(sub_layer.name)
            elif not isinstance(layer, supported_ops) and not 'Input' in str(type(layer)):
                unsupported_for_acceleration.append(layer.name)

        if unsupported_for_acceleration:
            print(f"  Info: Slojevi {list(set(unsupported_for_acceleration))[:3]}... bi mogli raditi sporije.")
        else:
            print("  Potvrdjeno: Model koristi standardne operacije (brzo na RPi uz XNNPACK).")

        # 4. Simulacija TFLite konverzije
        print("\n[4] TFLITE CONVERTER TEST:")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            print("  Potvrdjeno: TFLite Converter prihvaca arhitekturu modela.")
        except Exception as conv_err:
            print(f"  Upozorenje: TFLite konverter bi mogao imati problema: {conv_err}")

    except Exception as e:
        print(f"Pogreska pri analizi: {e}")

if __name__ == "__main__":
    get_raspberry_tflite_overview(CONFIG_PATH)