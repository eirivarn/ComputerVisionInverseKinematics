import cv2
import mediapipe as mp
from src.utils.hand_pose_utils import *


# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


def print_landmark_positions(image, results):
    desired_landmark_indexes = [0]

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            print(f'Hand {hand_index + 1}:')
            for landmark_index, landmark in enumerate(hand_landmarks.landmark):
                if landmark_index in desired_landmark_indexes:
                    image_height, image_width, _ = image.shape
                    x, y = int(
                        landmark.x * image_width), int(landmark.y * image_height)
                    print(
                        f'  Landmark {landmark_index}: (x: {x}, y: {y}, z: {landmark.z})')


def hand_tracking():
    cap = cv2.VideoCapture(0)
    middle_finger_counter = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hand_position_detected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

                image_height, image_width, _ = image.shape

                # Sjekk for spesifikke håndposisjoner
                if thumb_down(hand_landmarks):
                    print("Thumb down detected")
                    hand_position_detected = True
                if hand_open(hand_landmarks, image_width, image_height):
                    print("Hand open detected")
                    hand_position_detected = True
                if closed_fist(hand_landmarks, image_width, image_height, 100):
                    print("Closed fist detected")
                    hand_position_detected = True
                if middle_finger(hand_landmarks):
                    print("Middle finger detected")
                    middle_finger_counter += 1
                    hand_position_detected = True
                if not middle_finger(hand_landmarks):
                    middle_finger_counter = 0

        if not hand_position_detected:
            print("No hand gesture detected")

        cv2.imshow("Hand Tracking", image)
        # Avslutt hvis en håndposisjon er detektert eller ESC trykkes
        if middle_finger_counter > 10 or cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


hand_tracking()
