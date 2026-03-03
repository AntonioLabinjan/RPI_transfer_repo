import os
import sys
import cv2
import numpy as np
import threading
import time
import argparse
from scipy.spatial import distance as dist
from collections import deque

# --- 1. NUMPY PATCHES ---
if not hasattr(np, 'strings'):
    import types
    sd = types.ModuleType("strings")
    sd.encode = lambda array, encoding='utf-8': np.array([str(s).encode(encoding) for s in array]) if not np.isscalar(array) else np.array(str(array).encode(encoding))
    np.strings = sd
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'bool'): np.bool = bool

import tf_keras as keras
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm

# --- 2. KONFIGURACIJA ---
MODEL_CONFIG = "./outputs/best_person_detection (2)/config.json"
MODEL_WEIGHTS = "./outputs/best_person_detection (2)/model.weights.h5"
THRESHOLD = 0.53
IMG_SIZE = 320
SMOOTHING_WINDOW = 8

MIN_BLOB_AREA = 1200   
MAX_DISAPPEARED = 30 
MIN_DENSITY = 0.03     
MIN_WIDTH, MIN_HEIGHT = 45, 80 
MAX_ROI_FRAC = 0.8    

# Globalne varijable stanja
latest_frame = None
is_running = True
tracked_objects = {}  
next_object_id = 0
roi_mode_active = True  
global_label = "Scanning..."
global_prob = 0.0
global_history = deque(maxlen=SMOOTHING_WINDOW)

model_ready_event = threading.Event()

# --- 3. TRACKER LOGIKA ---
def update_tracker(new_rects):
    global tracked_objects, next_object_id
    if not roi_mode_active:
        if tracked_objects: tracked_objects.clear()
        return
    
    if new_rects is None or len(new_rects) == 0:
        for obj_id in list(tracked_objects.keys()):
            tracked_objects[obj_id]["disappeared"] += 1
            if tracked_objects[obj_id]["disappeared"] > MAX_DISAPPEARED:
                del tracked_objects[obj_id]
        return

    new_centroids = np.array([(x + w//2, y + h//2) for (x, y, w, h) in new_rects])
    
    if not tracked_objects:
        for i, rect in enumerate(new_rects):
            tracked_objects[next_object_id] = {
                "box": tuple(rect), "centroid": new_centroids[i], 
                "label": "Wait...", "prob": 0.0, "disappeared": 0,
                "history": deque(maxlen=SMOOTHING_WINDOW)
            }
            next_object_id += 1
    else:
        object_ids = list(tracked_objects.keys())
        object_centroids = np.array([obj["centroid"] for obj in tracked_objects.values()])
        D = dist.cdist(object_centroids, new_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_rows, used_cols = set(), set()
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols: continue
            obj_id = object_ids[row]
            tracked_objects[obj_id].update({"box": tuple(new_rects[col]), "centroid": new_centroids[col], "disappeared": 0})
            used_rows.add(row); used_cols.add(col)
            
        for col in range(len(new_centroids)):
            if col not in used_cols:
                tracked_objects[next_object_id] = {
                    "box": tuple(new_rects[col]), "centroid": new_centroids[col], 
                    "label": "Wait...", "prob": 0.0, "disappeared": 0,
                    "history": deque(maxlen=SMOOTHING_WINDOW)
                }
                next_object_id += 1
                
        for row in range(len(object_centroids)):
            if row not in used_rows:
                obj_id = object_ids[row]
                tracked_objects[obj_id]["disappeared"] += 1
                if tracked_objects[obj_id]["disappeared"] > MAX_DISAPPEARED:
                    del tracked_objects[obj_id]

# --- 4. AI THREAD ---
def ai_thread_worker(model_config, model_weights):
    global latest_frame, tracked_objects, is_running, global_label, global_prob, global_history
    print("🧠 [AI] Inicijalizacija modela...")
    try:
        with open(model_config, 'r') as f: model = keras.models.model_from_json(f.read())
        model.load_weights(model_weights)
        dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        print("✅ [AI] Model spreman.")
        model_ready_event.set() 
    except Exception as e:
        print(f"❌ [AI ERROR] {e}")
        is_running = False
        return

    while is_running:
        if latest_frame is not None:
            local_frame = latest_frame.copy()
            if roi_mode_active:
                for obj_id, data in list(tracked_objects.items()):
                    if data["disappeared"] > 0: continue
                    x, y, w, h = data["box"]
                    pw, ph = int(w * 0.15), int(h * 0.15)
                    crop = local_frame[max(0, y-ph):y+h+ph, max(0, x-pw):x+w+pw]
                    if crop.size == 0: continue
                    
                    img_in = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (IMG_SIZE, IMG_SIZE))
                    img_in = img_in.reshape((1, IMG_SIZE, IMG_SIZE, 1)).astype('float32') / 255.0
                    p = model.predict(img_in, verbose=0)[0][0]
                    
                    if obj_id in tracked_objects:
                        tracked_objects[obj_id]["history"].append(p)
                        avg_p = sum(tracked_objects[obj_id]["history"]) / len(tracked_objects[obj_id]["history"])
                        tracked_objects[obj_id]["prob"] = avg_p
                        tracked_objects[obj_id]["label"] = "HUMAN" if avg_p > THRESHOLD else "BG"
            else:
                img_in = cv2.resize(cv2.cvtColor(local_frame, cv2.COLOR_BGR2GRAY), (IMG_SIZE, IMG_SIZE))
                img_in = img_in.reshape((1, IMG_SIZE, IMG_SIZE, 1)).astype('float32') / 255.0
                p = model.predict(img_in, verbose=0)[0][0]
                global_history.append(p)
                global_prob = sum(global_history) / len(global_history)
                global_label = "HUMAN DETECTED" if global_prob > THRESHOLD else "NO HUMAN"
            time.sleep(0.01)
        else: time.sleep(0.1)

# --- 5. MAIN ---
def main():
    global latest_frame, is_running, tracked_objects, roi_mode_active

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="", help="Path to RAW/DAT or camera ID")
    parser.add_argument("--roi", type=str, default="ON")
    parser.add_argument("-l", "--loop", action="store_true", help="Loop the input file")
    args = parser.parse_args()

    roi_mode_active = True if args.roi.upper() == "ON" else False

    threading.Thread(target=ai_thread_worker, args=(MODEL_CONFIG, MODEL_WEIGHTS), daemon=True).start()
    print("⏳ Čekam AI model...")
    model_ready_event.wait()

    def on_frame_cb(ts, frame):
        global latest_frame, tracked_objects, roi_mode_active
        latest_frame = frame.copy()
        cv2.rectangle(frame, (0, 0), (320, 40), (20, 20, 20), -1)
        status_text = f"MODE: {'ROI' if roi_mode_active else 'GLOBAL'}"
        cv2.putText(frame, status_text, (10, 25), 0, 0.6, (255, 255, 255), 2)

        if roi_mode_active:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            clean_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)); merge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) 
            thresh_base = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_k); thresh_dilated = cv2.dilate(thresh_base, merge_k, iterations=3)
            contours, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            raw_rects = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if (w*h) < MIN_BLOB_AREA or w < MIN_WIDTH or h < MIN_HEIGHT: continue
                if (cv2.countNonZero(thresh_base[y:y+h, x:x+w]) / (w*h)) < MIN_DENSITY: continue
                raw_rects.append([x, y, w, h])

            grouped_rects, _ = cv2.groupRectangles(raw_rects + raw_rects, 1, 0.2) if raw_rects else ([], None)
            update_tracker(grouped_rects)

            for obj_id, data in tracked_objects.items():
                if data["disappeared"] > 2: continue
                x, y, w, h = data["box"]
                color = (0, 255, 0) if data["label"] == "HUMAN" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID:{obj_id} {data['label']} {data['prob']:.2f}", (x, y-10), 0, 0.5, color, 1)
        else:
            color = (0, 255, 0) if "DETECTED" in global_label else (0, 0, 255)
            cv2.putText(frame, f"STATE: {global_label}", (10, 85), 0, 0.8, color, 2)
        cv2.imshow("Detection System", frame)

    while is_running:
        try:
            mv_it = EventsIterator(input_path=args.input, delta_t=33000)
            ev_width, ev_height = mv_it.get_size()
            
            activity_filter = ActivityNoiseFilterAlgorithm(ev_width, ev_height, 20000)
            events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
            frame_gen = PeriodicFrameGenerationAlgorithm(ev_width, ev_height, fps=30)
            frame_gen.set_output_callback(on_frame_cb)

            print(f"🎬 Pokrećem feed: {args.input if args.input else 'Live Camera'}")
            
            # --- 1:1 SYNC LOGIKA ---
            start_time = time.perf_counter()
            start_ts = None

            for ev in mv_it:
                # Uzimamo trenutni timestamp iz event streama (u mikrosekundama)
                current_ts = mv_it.get_current_time()
                
                if start_ts is None:
                    start_ts = current_ts

                # Izračunavamo koliko je vremena prošlo u datoteci (pretvoreno u sekunde)
                file_elapsed = (current_ts - start_ts) / 1e6
                # Koliko je stvarno vremena prošlo od početka replaya
                real_elapsed = time.perf_counter() - start_time

                # Ako idemo prebrzo, pauziraj
                if file_elapsed > real_elapsed:
                    time.sleep(file_elapsed - real_elapsed)

                activity_filter.process_events(ev, events_buf)
                frame_gen.process_events(events_buf)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    is_running = False
                    break
                elif key == ord('r'):
                    roi_mode_active = not roi_mode_active
            
            if not args.loop or not is_running:
                break
            else:
                print("🔄 Restartam loop...")
                tracked_objects.clear()
                
        except Exception as e:
            print(f"❌ [GREŠKA] {e}")
            break

    is_running = False
    cv2.destroyAllWindows()

if __name__ == "__main__": main()