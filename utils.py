import numpy as np
import os

RAW_SAMPLES_DIR='samples/raw'
SCREEN_SAMPLES_DIR='samples/screen'

def get_files(input_dir):
    return [os.path.join(input_dir, file_name) for file_name in os.listdir(input_dir)]

dist = lambda p,q: np.linalg.norm(q-p)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
