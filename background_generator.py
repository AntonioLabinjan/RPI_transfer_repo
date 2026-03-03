import os
import cv2
import numpy as np
import time
from datetime import datetime

# --- 1. PATCHEVI ZA NUMPY (Zbog onog 2.0 kaosa) ---
if not hasattr(np, 'strings'):
    import types
    sd = types.ModuleType("strings")
    sd.encode = lambda array, encoding='utf-8': np.array([str(s).encode(encoding) for s in array]) if not np.isscalar(array) else np.array(str(array).encode(encoding))
    np.strings = sd
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'bool'): np.bool = bool

# --- 2. IMPORTI ---
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm

# --- 3. KONFIGURACIJA ---
SAVE_PATH = os.path.expanduser("~/Desktop/background")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

TARGET_SIZE = 320
AUTO_SAVE_INTERVAL = 2.0  # Sekunde između automatskog spremanja
ACTIVITY_THS = 20000

def resize_and_crop(img, size):
    """Smanjuje sliku i reže je na 320x320 iz sredine (proporcionalno)."""
    h, w = img.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    start_x = (new_w - size) // 2
    start_y = (new_h - size) // 2
    return img_resized[start_y:start_y+size, start_x:start_x+size]

def main():
    # 4. INICIJALIZACIJA KAMERE
    try:
        mv_it = EventsIterator(input_path="", delta_t=33000)
        ev_width, ev_height = mv_it.get_size()
    except Exception as e:
        print(f"❌ Kamera error: {e}")
        return

    # 5. ALGORITMI
    activity_filter = ActivityNoiseFilterAlgorithm(ev_width, ev_height, ACTIVITY_THS)
    events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    frame_gen = PeriodicFrameGenerationAlgorithm(ev_width, ev_height, fps=30)
    
    current_frame_320 = None
    last_save_time = time.time()

    def on_frame_cb(ts, frame):
        nonlocal current_frame_320
        current_frame_320 = resize_and_crop(frame, TARGET_SIZE)

    frame_gen.set_output_callback(on_frame_cb)

    print(f"🚀 Logger pokrenut! Slike idu u: {SAVE_PATH}")
    print("⌨️ Tipke: [SPACE] Spremi zadnji frame i UGASI program, [ESC] Izlaz bez spremanja")

    try:
        for ev in mv_it:
            activity_filter.process_events(ev, events_buf)
            frame_gen.process_events(events_buf)
            
            if current_frame_320 is not None:
                cv2.imshow("Dataset Collector (320x320)", current_frame_320)
                
                # --- AUTOMATSKO SPREMANJE ---
                now = time.time()
                if now - last_save_time >= AUTO_SAVE_INTERVAL:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    fname = os.path.join(SAVE_PATH, f"auto_{timestamp}.png")
                    cv2.imwrite(fname, current_frame_320)
                    last_save_time = now
                    print(f"💾 Auto-save: {fname}")

                # --- TIPKOVNICA ---
                key = cv2.waitKey(1) & 0xFF
                if key == 27: # ESC za izlaz
                    break
                elif key == 32: # SPACE za spremanje i GAŠENJE
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    fname = os.path.join(SAVE_PATH, f"final_{timestamp}.png")
                    cv2.imwrite(fname, current_frame_320)
                    print(f"📸 Spremljen zadnji frame: {fname}")
                    print("🛑 Automatsko gašenje...")
                    break # Izlaz iz for petlje

    finally:
        cv2.destroyAllWindows()
        print("👋 Kolektor ugašen.")

if __name__ == "__main__":
    main()