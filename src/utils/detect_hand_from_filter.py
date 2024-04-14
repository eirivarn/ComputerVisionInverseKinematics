import cv2
import numpy as np

# Function to do nothing on trackbar move, required by OpenCV
def nothing(x):
    """Dummy function for OpenCV trackbar callbacks."""
    pass

# Function to create trackbars for HSV tuning
def create_hsv_trackbars(window_name):
    """
    Create HSV tuning trackbars in the specified window.
    
    Args:
    window_name (str): The name of the window where the trackbars will be displayed.
    """
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
    """
    Apply HSV filtering to a frame based on trackbar positions.
    
    Args:
    frame (np.array): The frame to filter.
    window_name (str): The window where the trackbars are located.
    
    Returns:
    np.array: The filtered frame.
    """
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



def estimate_hand_open_or_closed(contour):
    """
    Estimate if the hand represented by a contour is open or closed.
    
    Args:
    contour (np.array): The contour to evaluate.
    
    Returns:
    str: "Open" if the hand is open, "closed" otherwise.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    (x, y), (width, height), angle = rect
    
    aspect_ratio = max(width, height) / min(width, height) # Aspect ratio is the ratio of width to height
    area_contour = cv2.contourArea(contour)
    area_bbox = width * height
    solidity = area_contour / area_bbox # Solidity is the ratio of contour area to bounding box area

    
    if aspect_ratio >= 1.7 and solidity < 0.55:  
        return "Open"
    return "closed"

# Initialize last known centroid
last_centroid = None

def process_contours(contours, frame, window_name):
    """
    Process detected contours to find and draw the largest based on the previous centroid.

    Args:
    contours (list): List of detected contours.
    frame (np.array): The current video frame.
    window_name (str): The window where trackbars are displayed.

    Returns:
    tuple: (centroid, contour) if a suitable contour is found, (None, None) otherwise.
    """
    global last_centroid
    min_area = cv2.getTrackbarPos('Min Contour Area', window_name)
    max_area = cv2.getTrackbarPos('Max Contour Area', window_name)
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

    if not valid_contours:
        return None, None  # Return None, None if no valid contours found

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
            return (cX, cY), largest_contour

    return None, None
