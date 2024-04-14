from roboticstoolbox import DHRobot, RevoluteDH
import numpy as np
import cv2
# import mediapipe as mp
from matplotlib import pyplot as plt 
import matplotlib.patches as patches


# def get_hand_position(hand_landmarks, image_width, image_height):
#     wrist_landmark = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
#     x = wrist_landmark.x * image_width
#     y = image_height - (wrist_landmark.y * image_height)
#     return np.array([x, y])


def calculate_inverse_kinematics(target_pos_px, image_width, image_height):
    # Konverter pikselposisjon til "meter" i robotens koordinatsystem
    scale_x, scale_y = 2.0 / image_width, 2.0 / image_height
    x, y = target_pos_px[0] * scale_x - 1, target_pos_px[1] * \
        scale_y - 1  # Juster slik at (0,0) er i midten

    # Beregn invers kinematikk (forenklet for 2D planar robot)
    a1, a2 = 0.5, 0.5  # Armens lengder
    d = np.sqrt(x**2 + y**2)

    if d > (a1 + a2):
        print("Målet er utenfor rekkevidde.")
        return None

    cos_theta2 = (d**2 - a1**2 - a2**2) / (2 * a1 * a2)
    theta2 = np.arccos(cos_theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(a2 *
                                           np.sin(theta2), a1 + a2 * np.cos(theta2))

    return np.array([theta1, theta2])


# # Initialiser MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
#                        min_detection_confidence=0.5, min_tracking_confidence=0.5)


# # Forbered plotting
# fig, ax = plt.subplots()
# plt.ion()  # Slå på interaktiv modus for sanntidsplotting


import matplotlib.patches as patches

def draw_robot_arm(q, ax, end_effector_closed):
    a1, a2 = 0.5, 0.5  # Arm lengths
    # Calculate the positions of the first and second joint based on joint angles
    joint1 = (a1 * np.cos(q[0]), a1 * np.sin(q[0]))
    joint2 = (joint1[0] + a2 * np.cos(q[0] + q[1]), joint1[1] + a2 * np.sin(q[0] + q[1]))

    # Clear previous drawing
    # ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Draw the robot arm: base -> first joint -> second joint
    ax.plot([0, joint1[0]], [0, joint1[1]], 'r-', lw=4)  # From base to first joint
    ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], 'b-', lw=4)  # From first joint to second joint
    ax.plot(joint1[0], joint1[1], 'ko', markersize=5)  # Pivot point

    # End effector enhancements
    effector_length = 0.15  # Length of the end effector 'jaws'
    angle_offset = np.pi / 6  # Angle for the cooler look
    pinch_angle = np.pi / 6  # Constant angle for the pinching joint
    joint_angle = q[0] + q[1]  # Total angle of the end effector

    # Calculate positions for the end effector 'jaws'
    if end_effector_closed:
        left_jaw = (joint2[0] + effector_length * np.cos(joint_angle - angle_offset),
                    joint2[1] + effector_length * np.sin(joint_angle - angle_offset))
        right_jaw = (joint2[0] + effector_length * np.cos(joint_angle + angle_offset),
                     joint2[1] + effector_length * np.sin(joint_angle + angle_offset))
    else:
        left_jaw = (joint2[0] + effector_length * np.cos(joint_angle - 2*angle_offset),
                    joint2[1] + effector_length * np.sin(joint_angle - 2*angle_offset))
        right_jaw = (joint2[0] + effector_length * np.cos(joint_angle + 2*angle_offset),
                     joint2[1] + effector_length * np.sin(joint_angle + 2*angle_offset))

    # Calculate pinching joint positions
    left_pinch = (left_jaw[0] + effector_length * 0.5 * np.cos(np.pi/2 + joint_angle - angle_offset - pinch_angle),
                  left_jaw[1] + effector_length * 0.5 * np.sin(np.pi/2 + joint_angle - angle_offset - pinch_angle))
    right_pinch = (right_jaw[0] + effector_length * 0.5 * np.cos(-np.pi/2 + joint_angle + angle_offset + pinch_angle),
                   right_jaw[1] + effector_length * 0.5 * np.sin(-np.pi/2 + joint_angle + angle_offset + pinch_angle))
    
    # Draw the end effector and pinching joints
    ax.plot([joint2[0], left_jaw[0]], [joint2[1], left_jaw[1]], 'k-', lw=2)  # Left 'jaw'
    ax.plot([joint2[0], right_jaw[0]], [joint2[1], right_jaw[1]], 'k-', lw=2)  # Right 'jaw'
    ax.plot([left_jaw[0], left_pinch[0]], [left_jaw[1], left_pinch[1]], 'g-', lw=2)  # Left pinching joint
    ax.plot([right_jaw[0], right_pinch[0]], [right_jaw[1], right_pinch[1]], 'g-', lw=2)  # Right pinching joint

    plt.pause(0.001)  # A short pause to ensure the plot updates



def hand_tracking_and_control_robot():
    end_effector_closed = False  
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_pos_px = get_hand_position(hand_landmarks, image.shape[1], image.shape[0])
                q = calculate_inverse_kinematics(hand_pos_px, image.shape[1], image.shape[0])
                if q is not None:
                    # Toggle the state of the end effector for demonstration
                    end_effector_closed = not end_effector_closed  # Toggle state
                    # Draw the robot arm based on calculated joint angles and end effector state
                    draw_robot_arm(q, ax, end_effector_closed)

        cv2.imshow("Hand Tracking", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  


hand_tracking_and_control_robot()

