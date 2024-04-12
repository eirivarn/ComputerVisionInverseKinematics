import cv2
import numpy as np
from src.utils.inverse_kinematics import calculate_inverse_kinematics, draw_robot_arm
from src.utils.detect_hand_from_filter import apply_hsv_filter, image_processing, process_contours, create_hsv_trackbars

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Create a window with HSV trackbars
    cv2.namedWindow('HSV Tuner')
    create_hsv_trackbars('HSV Tuner')

    # Initialize background subtractor for moving object detection
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply HSV filter
        filtered_frame = apply_hsv_filter(frame, 'HSV Tuner')
        cv2.imshow('Filtered', filtered_frame)

        # Process image to find contours
        thresh = image_processing(filtered_frame, fgbg)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours to find the largest one and get its centroid
        centroid = process_contours(contours, frame, 'HSV Tuner')
        if centroid:
            # Adjust the centroid before kinematic calculations:
            # Here, as an example, we adjust y to increase upwards:
            adjusted_centroid = np.array([centroid[0], frame.shape[0] - centroid[1]])

            # Calculate robot arm positions based on the adjusted centroid
            q = calculate_inverse_kinematics(adjusted_centroid, frame.shape[1], frame.shape[0])
            if q is not None:
                # Assuming end effector state is dynamic or set elsewhere; here we assume it's open
                draw_robot_arm(frame, q, end_effector_closed=False)

        # Display the original frame with any overlays from draw_robot_arm
        cv2.imshow('Original', frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()