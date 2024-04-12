import cv2
import numpy as np

# Function to do nothing on trackbar move, required by OpenCV
def nothing(x):
    pass

# Function to create trackbars for HSV tuning
def create_hsv_trackbars(window_name):
    cv2.createTrackbar('H Lower', window_name, 0, 179, nothing)
    cv2.createTrackbar('H Upper', window_name, 179, 179, nothing)
    cv2.createTrackbar('S Lower', window_name, 0, 255, nothing)
    cv2.createTrackbar('S Upper', window_name, 255, 255, nothing)
    cv2.createTrackbar('V Lower', window_name, 0, 255, nothing)
    cv2.createTrackbar('V Upper', window_name, 255, 255, nothing)
    cv2.createTrackbar('Min Contour Area', window_name, 0, 5000, nothing)  
    cv2.createTrackbar('Max Contour Area', window_name, 5000, 1000000, nothing)  

# Function to get current HSV filter settings from the trackbars
def apply_hsv_filter(frame, window_name):
    lower_hsv = np.array([cv2.getTrackbarPos('H Lower', window_name),
                          cv2.getTrackbarPos('S Lower', window_name),
                          cv2.getTrackbarPos('V Lower', window_name)])
    upper_hsv = np.array([cv2.getTrackbarPos('H Upper', window_name),
                          cv2.getTrackbarPos('S Upper', window_name),
                          cv2.getTrackbarPos('V Upper', window_name)])
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    # Improve mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_and(frame, frame, mask=mask)

def image_processing(filtered_frame, fgbg):
    gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    _, thresh = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)
    return cv2.GaussianBlur(thresh, (5, 5), 0)

# Initialize last known centroid
last_centroid = None

def process_contours(contours, frame, window_name):
    global last_centroid
    min_area = cv2.getTrackbarPos('Min Contour Area', window_name)
    max_area = cv2.getTrackbarPos('Max Contour Area', window_name)

    # Filter contours based on the area
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    if not valid_contours:
        return None  # Return None if no valid contours found

    largest_contour = None
    # Calculate distances from last_centroid for each valid contour
    if last_centroid is not None:
        contours_distances = []
        for c in valid_contours:
            M = cv2.moments(c)  # Calculate moments for each contour
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = np.array([cx, cy])
                distance = np.linalg.norm(last_centroid - centroid)
                contours_distances.append((distance, c))

        # Find the contour with the minimum distance to last_centroid
        largest_contour = min(contours_distances, key=lambda x: x[0])[1] if contours_distances else None
    else:
        # Just find the largest contour by area
        largest_contour = max(valid_contours, key=cv2.contourArea)

    # If a suitable contour is found, update last_centroid and return its center
    if largest_contour is not None:
        cv2.drawContours(frame, [largest_contour], -1, (50, 255, 0), 3)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            last_centroid = np.array([cX, cY])
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            return (cX, cY)  # Return the centroid coordinates

    return None  # Return None if no contour met the criteria


def detect_largest_moving_contour_with_tuning():
    global last_centroid
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video source.")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    cv2.namedWindow('HSV Tuner')
    create_hsv_trackbars('HSV Tuner')

    while True:
        ret, frame = video.read()
        if not ret:
            break

        filtered_frame = apply_hsv_filter(frame, 'HSV Tuner')
        cv2.imshow('Filtered', filtered_frame)

        thresh = image_processing(filtered_frame, fgbg)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        process_contours(contours, frame, 'HSV Tuner')

        cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()

