import cv2
import mediapipe as mp
from src.utils.hand_pose_utils import thumb_down, hand_open, closed_fist, middle_finger
from src.utils.inverse_kinematics import calculate_inverse_kinematics, draw_robot_arm, get_hand_position

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)


def hand_tracking_and_control_robot_combined():
    """
    Main function to control video capture and image processing for hand detection in real-time.
    
    Initiates video capture, sets up a GUI for HSV tuning, and processes the video stream to:
    - Filter the image based on HSV values adjustable via trackbars.
    - Detect contours of a hand in the video.
    - Determine the hand's status (open/closed) based on the largest detected contour.
    - Display the status on the frame.
    - Adjust calculations for robot arm positioning if a suitable contour is detected.
    - Display both the original and the processed frames until the user exits with a 'q' key press.
    
    The process involves:
    - Video capture setup and validation.
    - Creation of an OpenCV window with interactive HSV trackbars.
    - Frame-by-frame image processing:
      1. Applying an HSV filter.
      2. Converting the image to grayscale and thresholding.
      3. Contour detection and processing.
      4. Robot arm position calculation and drawing based on the hand's detected position.
    - Continuous display updates and exit upon user command.
    """
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    middle_finger_counter = 0
    while cap.isOpened():
        
        success, image = cap.read()
        if not success:
            continue
        
        # Convert image from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image with mediapipe Hands
        results = hands.process(image)  # Make sure 'hands' is initialized correctly before this line
        
        # Convert image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hand_position_detected = False
        end_effector_closed = True
        
        # Draw the hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

                # Get the image dimensions
                image_height, image_width, _ = image.shape

                # Check for different hand gestures
                if thumb_down(hand_landmarks):
                    print("Thumb down detected")
                    hand_position_detected = True
                if hand_open(hand_landmarks, image_width, image_height):
                    print("Hand open detected")
                    hand_position_detected = True
                    end_effector_closed = False
                if closed_fist(hand_landmarks, image_width, image_height, 100):
                    print("Closed fist detected")
                    hand_position_detected = True
                    end_effector_closed = True
                if middle_finger(hand_landmarks):
                    print("Middle finger detected")
                    middle_finger_counter += 1
                    hand_position_detected = True
                if not middle_finger(hand_landmarks):
                    middle_finger_counter = 0

                # Calculate the robot arm position based on the hand position
                hand_pos_px = get_hand_position(hand_landmarks, image.shape[1], image.shape[0])
                q = calculate_inverse_kinematics(hand_pos_px, image.shape[1], image.shape[0])
                if q is not None:
                    image = draw_robot_arm(image, q, end_effector_closed)  # Draw the robot arm on the image
        if not hand_position_detected:
            print("No hand gesture detected")
        
        cv2.imshow("Hand Tracking", image)
        if middle_finger_counter > 10 or cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    hand_tracking_and_control_robot_combined()
