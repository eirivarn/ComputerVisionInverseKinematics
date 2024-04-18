import numpy as np
import cv2
import mediapipe as mp

def get_hand_position(hand_landmarks, image_width, image_height):
    """
    Calculate the pixel position of the wrist landmark on an image.
    
    Args:
    hand_landmarks (mediapipe.HandLandmarks): Detected hand landmarks.
    image_width (int): The width of the image where landmarks were detected.
    image_height (int): The height of the image where landmarks were detected.
    
    Returns:
    np.array: The (x, y) coordinates of the wrist landmark.
    """
    wrist_landmark = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
    x = wrist_landmark.x * image_width
    y = image_height + (wrist_landmark.y * image_height)
    return np.array([x, y])


def calculate_inverse_kinematics(target_pos_px, image_width, image_height):
    """
    Convert a pixel position to "meters" in the robot coordinate system and calculate inverse kinematics.

    Args:
    target_pos_px (np.array): The target position in pixel coordinates.
    image_width (int): Width of the image to scale the position.
    image_height (int): Height of the image to scale the position.

    Returns:
    np.array: Joint angles (theta1, theta2) in radians, or None if the target is out of reach.
    """
    # Convert pixel position to normalized device coordinates in the range [-1, 1]
    scale_x, scale_y = 2.0 / image_width, 2.0 / image_height
    x, y = target_pos_px[0] * scale_x - 1, target_pos_px[1] * \
        scale_y - 1  # Adjust so that (0, 0) is in the center
        
    a1, a2 = 0.5, 0.5  # Arm lengths in meters
    d = np.sqrt(x**2 + y**2) # Distance from the base to the target position

    # Check if the target is reachable
    if d > (a1 + a2):
        print("The target position is out of reach for the robot arm.")
        return None

    # Calculate joint angles using cosine law
    cos_theta2 = (d**2 - a1**2 - a2**2) / (2 * a1 * a2)
    theta2 = np.arccos(cos_theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(a2 *
                                           np.sin(theta2), a1 + a2 * np.cos(theta2))

    return np.array([theta1, theta2])


def draw_robot_arm(image, q, end_effector_closed):
    """
    Draw a simplistic robot arm on an image.
    
    Args:
    image (np.array): The image to draw the robot arm on.
    q (np.array): The joint angles (theta1, theta2) in radians.
    end_effector_closed (bool): Whether the end effector (gripper) is closed.

    Returns:
    np.array: The image with the robot arm drawn on it.
    """
    print(image.shape)
    a1, a2 = 200, 200  # Arm lengths in pixels
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2  # Center of the image

    # Calculate joint positions
    joint1 = (int(center_x + a1 * np.cos(q[0])), int(center_y - a1 * np.sin(q[0])))
    joint2 = (int(joint1[0] + a2 * np.cos(q[0] + q[1])), int(joint1[1] - a2 * np.sin(q[0] + q[1])))

    # Draw the robot arm
    cv2.line(image, (center_x, center_y), joint1, (0, 0, 255), 3)  # Base to first joint
    cv2.line(image, joint1, joint2, (255, 0, 0), 3)  # First joint to second joint
    cv2.circle(image, joint1, 5, (0, 255, 0), -1)  # First joint
    cv2.circle(image, joint2, 5, (0, 255, 0), -1)  # Second joint

    # Draw end-effector based on hand open/closed state
    effector_length = 40
    angle_offset = 0.3  # Radians for the gripper opening
    if end_effector_closed:
        angle_offset = 0  # No offset, closed gripper

    # Positions for the gripper
    gripper_left = (int(joint2[0] + effector_length * np.cos(q[0] + q[1] - angle_offset)),
                    int(joint2[1] - effector_length * np.sin(q[0] + q[1] - angle_offset)))
    gripper_right = (int(joint2[0] + effector_length * np.cos(q[0] + q[1] + angle_offset)),
                     int(joint2[1] - effector_length * np.sin(q[0] + q[1] + angle_offset)))

    # Draw gripper
    cv2.line(image, joint2, gripper_left, (0, 255, 0), 3)
    cv2.line(image, joint2, gripper_right, (0, 255, 0), 3)

    return image


# Function only used for testing, REMEMBER TO ALWAYS REMOVE CALLS TO THIS FUNCTION IN THIS FILE
def hand_tracking_and_control_robot():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    end_effector_closed = False
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_pos_px = get_hand_position(hand_landmarks, image.shape[1], image.shape[0])
                q = calculate_inverse_kinematics(hand_pos_px, image.shape[1], image.shape[0])
                if q is not None:
                    image = draw_robot_arm(image, q, end_effector_closed)  # Draw the robot arm on the image

        # Convert back to BGR for displaying
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

