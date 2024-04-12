import cv2
import numpy as np
from detect_hand_from_filter import apply_hsv_filter, create_hsv_trackbars

def estimate_hand_open_or_closed(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    (x, y), (width, height), angle = rect
    
    # Calculate aspect ratio and solidity
    aspect_ratio = max(width, height) / min(width, height)
    area_contour = cv2.contourArea(contour)
    area_bbox = width * height
    solidity = area_contour / area_bbox

    # Define more refined thresholds or conditions
    if aspect_ratio >= 1.7 and solidity < 0.55:  # Considered open if elongated and not very solid
        return "Open"
    return "closed"

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('HSV Tuner')
    create_hsv_trackbars('HSV Tuner')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filtered_frame = apply_hsv_filter(frame, 'HSV Tuner')
        gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hand_status = estimate_hand_open_or_closed(largest_contour)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
            cv2.putText(frame, 'Hand status: ' + hand_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Filtered Hand View', filtered_frame)
        cv2.imshow('Original View', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()