# camera
# CAMERA_URL = ['rtsp://admin:sys123456@185.123.3.152:554/cam/realmonitor?channel=1&subtype=0']
CAMERA_URL = ['../video/0.avi']

# roi
CAMERA_ROI = [[0.35, 0.18, 0.7, 0.52]]
CHECK_LINE = [[0.41, 0.045, 0.8, 0.5]]
DIRECTION = [{'start': [0.55, 0.1], 'end': [0.35, 0.1]}]
# CAMERA_ROI = [[0.25, 0.13, 0.7, 0.55]]
# CHECK_LINE = [[0.25, 0.3, 0.7, 0.3]]
# DIRECTION = [{'start': [0.3, 0.2], 'end': [0.3, 0.4]}]

# engine
RUN_MODE_THREAD = True
RESIZE_FACTOR = 0.5
DISPLAY_DETECT_FRAME_ONLY = True
DISPLAY_TRACK_INDEX = False
SHOW_VIDEO = True
SAVE_VIDEO = True
SAVE_CROP = True
SAVE_PERIOD = 1800

# detector
DETECT_ENABLE = True
DETECT_GROUP = False
USING_SERVICE = False
# SERVICE_URL = 'http://localhost:30001/head_detection/v1.0'
SERVICE_URL = 'http://52.142.14.71:3000/head_detection/v1.0'
DETECTION_THRESHOLD = 0.1
SIZE_THRESHOLD = 90

# tracker
TRACKER_THRESHOLD_DISTANCE = 100  # 90
TRACKER_BUFFER_LENGTH = 100
TRACKER_KEEP_LENGTH = 5
DISTANCE_MARGIN_IN = 20
DISTANCE_MARGIN_OUT = -10
DISTANCE_THRESHOLD = 50
