import cv2
import numpy as np
from src.utils.inverse_kinematics import calculate_inverse_kinematics, draw_robot_arm
from src.utils.detect_hand_from_filter import apply_hsv_filter, process_contours, create_hsv_trackbars, estimate_hand_open_or_closed

def main():
    """
    Main function to process video feed for hand detection and control a simulated robot arm.
    
    Initializes video capture and creates a GUI for HSV thresholding. Processes each video frame
    to detect hand contours, determine hand open/closed status, and calculate the position of a
    robot arm based on the hand's position. The robot arm is then drawn on the frame.
    
    Uses:
    - OpenCV for video capture and image processing.
    - NumPy for numerical operations.
    
    Flow:
    1. Start video capture.
    2. Create an OpenCV window with HSV trackbars for live tuning.
    3. Process each frame to:
       a. Apply an HSV filter.
       b. Convert to grayscale and threshold.
       c. Detect and process contours.
       d. Calculate the position for the robot arm.
       e. Draw the robot arm on the frame.
    4. Display the original and processed frames.
    5. End capture on pressing the 'q' key.
    """
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Create a window with HSV trackbars
    cv2.namedWindow('HSV Tuner')
    create_hsv_trackbars('HSV Tuner')

    # Initialize last known centroid
    global last_centroid
    last_centroid = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply HSV filter
        filtered_frame = apply_hsv_filter(frame, 'HSV Tuner')
        cv2.imshow('Filtered', filtered_frame)
        
        # Convert to grayscale and apply thresholding
        gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to find the largest one and get its centroid
        centroid, contour = process_contours(contours, frame, 'HSV Tuner')
        if centroid:
            hand_status = estimate_hand_open_or_closed(contour)
            if hand_status == "Open":
                hand_closed = False
            else:
                hand_closed = True
            cv2.putText(frame, 'Hand status: ' + hand_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Adjust centroid to match the robot arm coordinate system
            adjusted_centroid = np.array([centroid[0], frame.shape[0] - centroid[1]])

            # Calculate robot arm positions based on the adjusted centroid
            q = calculate_inverse_kinematics(adjusted_centroid, frame.shape[1], frame.shape[0])
            if q is not None:
                # Assuming end effector state is dynamic or set elsewhere; here we assume it's open
                draw_robot_arm(frame, q, end_effector_closed=hand_closed)
            
        # Display the original frame with any overlays from draw_robot_arm
        cv2.imshow('Original', frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()