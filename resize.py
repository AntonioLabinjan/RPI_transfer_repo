# THIS STEP RESIZES IMAGE TO 320x320
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.patches import Rectangle
import PIL.Image as Image

# --- CONFIGURATION ---
dataset_numpy_path = "./examples/numpy/train"
dataset_xml_path = "./examples/xml/train"
target_res = 320  # New resolution: 320x320

# Get all .npy files in the folder and sort them
event_files = sorted(glob.glob(os.path.join(dataset_numpy_path, "*.npy")))

# --- ITERATION LOOP ---
for file_path in event_files:
    frame_name = os.path.basename(file_path).replace(".npy", "")
    print(f"Processing {frame_name}...")

    # 1. BBOX / XML Parsing
    xml_file = os.path.join(dataset_xml_path, f"{frame_name}.xml")
    if not os.path.exists(xml_file):
        print(f"Warning: XML for {frame_name} not found. Skipping.")
        continue

    tree = ET.parse(xml_file) 
    root = tree.getroot()
    
    # Original dimensions from XML
    orig_w = int(root.find("size")[0].text)
    orig_h = int(root.find("size")[1].text)
    
    # Calculate scaling factors
    scale_x = target_res / orig_w
    scale_y = target_res / orig_h
    
    bbox_coordinates = []
    for member in root.findall('object'):
        class_name = member[0].text
        # Scale bounding box coordinates to 320x320
        xmin = int(int(member[4][0].text) * scale_x)
        ymin = int(int(member[4][1].text) * scale_y)
        xmax = int(int(member[4][2].text) * scale_x)
        ymax = int(int(member[4][3].text) * scale_y)
        bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])

    # 2. SAE Processing
    events = np.load(file_path)
    # Load events and immediately scale X and Y coordinates
    df_events = pd.DataFrame({
        'timestamp': events[:,0], 
        'x': (events[:,1] * scale_x).astype(int), 
        'y': (events[:,2] * scale_y).astype(int), 
        'polarity': events[:,3]
    })

    # Filter events to ensure they stay within the 320x320 bounds
    df_events = df_events[(df_events['x'] < target_res) & (df_events['y'] < target_res)]

    # Time logic
    time_interval = 40e3
    timestamps_vector = df_events['timestamp'].to_numpy()
    if len(timestamps_vector) == 0: continue 
    
    time_limit = int(timestamps_vector[-1])
    t_init_0 = int(time_limit - time_interval)

    # Process surfaces into a 320x320 array
    sae = np.zeros((target_res, target_res, 2), dtype='float32')

    for pol, channel in [(0, 1), (1, 0)]:
        subset = df_events[df_events['polarity'] == pol].sort_values('timestamp').drop_duplicates(['x', 'y'], keep='last')
        subset = subset[subset['timestamp'].between(t_init_0, time_limit)]
        
        x = subset['x'].to_numpy()
        y = subset['y'].to_numpy()
        t = subset['timestamp'].to_numpy()
        
        sae[y, x, channel] = (255 * ((t - t_init_0) / time_interval)).astype(int)

    # 3. Visualization
    # The resulting image will now be 320x320
    im = Image.fromarray(0.5 * sae[:,:,0] + 0.5 * sae[:,:,1]).convert("L")
    
    plt.figure(figsize=(6, 6)) # Adjusted figsize for square aspect ratio
    plt.imshow(im, cmap="gray", extent=[0, target_res, target_res, 0])
    ax = plt.gca()
    
    for bbox in bbox_coordinates:
        name, xmin, ymin, xmax, ymax = bbox
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        plt.text(xmin, ymin-2, name, color='green', fontsize=8)

    plt.title(f"SAE (320x320) - {frame_name}")
    plt.show()
    
    # break # Uncomment to test only one frame