import cv2
import numpy as np

def run_hand_color_tuner():
    def nothing(x):
        # Callback function for trackbar, does nothing but necessary for trackbar
        pass

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # Prepare window with trackbars
    cv2.namedWindow('HSV Tuner')
    cv2.createTrackbar('H Lower', 'HSV Tuner', 0, 179, nothing)
    cv2.createTrackbar('H Upper', 'HSV Tuner', 179, 179, nothing)
    cv2.createTrackbar('S Lower', 'HSV Tuner', 0, 255, nothing)
    cv2.createTrackbar('S Upper', 'HSV Tuner', 255, 255, nothing)
    cv2.createTrackbar('V Lower', 'HSV Tuner', 0, 255, nothing)
    cv2.createTrackbar('V Upper', 'HSV Tuner', 255, 255, nothing)

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read trackbar positions for all
        h_lower = cv2.getTrackbarPos('H Lower', 'HSV Tuner')
        h_upper = cv2.getTrackbarPos('H Upper', 'HSV Tuner')
        s_lower = cv2.getTrackbarPos('S Lower', 'HSV Tuner')
        s_upper = cv2.getTrackbarPos('S Upper', 'HSV Tuner')
        v_lower = cv2.getTrackbarPos('V Lower', 'HSV Tuner')
        v_upper = cv2.getTrackbarPos('V Upper', 'HSV Tuner')

        # Create the HSV mask
        lower_hsv = np.array([h_lower, s_lower, v_lower])
        upper_hsv = np.array([h_upper, s_upper, v_upper])
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Apply mask to the original frame
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Display the original and masked images
        cv2.imshow('Original', frame)
        cv2.imshow('Masked', result)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Now you can call this function to start the hand color tuner
run_hand_color_tuner()
