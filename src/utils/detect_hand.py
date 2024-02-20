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

# Function to get current HSV filter settings from the trackbars
def apply_hsv_filter(frame, window_name):
    # Fetch the current HSV filter settings from the trackbars
    lower_hsv = np.array([cv2.getTrackbarPos('H Lower', window_name),
                          cv2.getTrackbarPos('S Lower', window_name),
                          cv2.getTrackbarPos('V Lower', window_name)])
    upper_hsv = np.array([cv2.getTrackbarPos('H Upper', window_name),
                          cv2.getTrackbarPos('S Upper', window_name),
                          cv2.getTrackbarPos('V Upper', window_name)])
    # Apply the HSV filtering
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    return cv2.bitwise_and(frame, frame, mask=mask)

# Convert filtered frame to grayscale and apply background subtraction
def image_processing(filtred_frame, fgbg):
    gray = cv2.cvtColor(filtred_frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    _, thresh = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)
    return cv2.GaussianBlur(thresh, (5, 5), 0)
    
#  Process contours if any are found
def process_contours(contours, frame):
    if contours:
        # Determine the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [largest_contour], -1, (50, 255, 0), 3)

        # Calculate and draw the centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        
    
# Main function to detect and track the largest moving contour with live HSV tuning
def detect_largest_moving_contour_with_tuning():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video source.")
        return

    fgbg = cv2.createBackgroundSubtractorMOG2()

    cv2.namedWindow('HSV Tuner')
    create_hsv_trackbars('HSV Tuner')

    while True:
        ret, frame = video.read()
        if not ret:
            break

        filtered_frame = apply_hsv_filter(frame, 'HSV Tuner')
        cv2.imshow('Filtered', filtered_frame)

        # Convert filtered frame to grayscale and apply background subtraction
        thresh = image_processing(filtered_frame, fgbg)

        # Find contours from the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours if any are found
        process_contours(contours, frame)
        
        cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_largest_moving_contour_with_tuning()