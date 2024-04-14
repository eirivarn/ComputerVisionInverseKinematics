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

# Initialize last known centroid
last_centroid = None

def estimate_hand_open_or_closed(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    (x, y), (width, height), angle = rect
    
    
    aspect_ratio = max(width, height) / min(width, height)
    area_contour = cv2.contourArea(contour)
    area_bbox = width * height
    solidity = area_contour / area_bbox

    
    if aspect_ratio >= 1.7 and solidity < 0.55:  
        return "Open"
    return "closed"

def process_contours(contours, frame, window_name):
    global last_centroid
    min_area = cv2.getTrackbarPos('Min Contour Area', window_name)
    max_area = cv2.getTrackbarPos('Max Contour Area', window_name)
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

    if not valid_contours:
        return None

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
        hand_state = estimate_hand_open_or_closed(largest_contour)
        cv2.drawContours(frame, [largest_contour], -1, (50, 255, 0), 3)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            last_centroid = np.array([cX, cY])
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            return (cX, cY), largest_contour

    return None  # Return None if no contour met the criteria


def detect_largest_moving_contour_with_tuning():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('HSV Tuner')
    create_hsv_trackbars('HSV Tuner')
    global last_centroid
    last_centroid = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filtered_frame = apply_hsv_filter(frame, 'HSV Tuner')
        gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            centroid, hand_status = process_contours(contours, frame, 'HSV Tuner')
            if centroid:
                cv2.circle(frame, tuple(centroid), 5, (0, 0, 255), -1)

        cv2.imshow('Filtered Hand View', filtered_frame)
        cv2.imshow('Original View', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
