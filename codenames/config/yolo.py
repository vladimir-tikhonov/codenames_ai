from pathlib import Path

MODEL_PATH = Path('models/yolo/yolo_v2.h5')
MODEL_URL = 'https://www.dropbox.com/s/obn7woabs8wav11/yolo_v2.h5.zip?raw=1'
INPUT_SIZE = 416
GRID_SIZE = 13
MAX_BOXES = 25
IOU_THRESHOLD = 0.3
SCORE_THRESHOLD = 0.5
DETECTION_THRESHOLD = 0.3
ANCHORS = [4.05402, 1.71911, 1.92065, 2.83549, 2.86613, 1.26893, 4.70233, 2.28580, 3.44187, 2.17561]
