import numpy as np
import mediapipe as mp
"""
Hand Landmarks Reference:

0. Wrist
1. Thumb CMC (Carpometacarpal Joint)
2. Thumb MCP (Metacarpophalangeal Joint)
3. Thumb IP (Interphalangeal Joint)
4. Thumb Tip
5. Index Finger MCP
6. Index Finger PIP (Proximal Interphalangeal Joint)
7. Index Finger DIP (Distal Interphalangeal Joint)
8. Index Finger Tip
9. Middle Finger MCP
10. Middle Finger PIP
11. Middle Finger DIP
12. Middle Finger Tip
13. Ring Finger MCP
14. Ring Finger PIP
15. Ring Finger DIP
16. Ring Finger Tip
17. Pinky Finger MCP
18. Pinky Finger PIP
19. Pinky Finger DIP
20. Pinky Finger Tip

Each landmark is represented by an (x, y, z) coordinate, 
where x and y are normalized to [0.0, 1.0] by the image width and height respectively,
and z represents the landmark depth with the depth at the wrist being the origin, 
and the smaller values meaning closer to the camera.
"""


def calculate_distance(landmark1, landmark2, image_width, image_height):
    x1, y1 = int(landmark1.x * image_width), int(landmark1.y * image_height)
    x2, y2 = int(landmark2.x * image_width), int(landmark2.y * image_height)
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def calculate_angle(landmark1, landmark2, landmark3):
    a = np.array([landmark1.x - landmark2.x, landmark1.y - landmark2.y])
    b = np.array([landmark3.x - landmark2.x, landmark3.y - landmark2.y])
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def thumb_down(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

    thumb_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP]
    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

    return thumb_tip.y > thumb_mcp.y > wrist.y


# Funker ikke?????
def thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP]
    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

    return thumb_tip.y < thumb_mcp.y and thumb_tip.y < wrist.y


def hand_open(hand_landmarks, image_width, image_height):

    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    distance = calculate_distance(
        thumb_tip, index_finger_tip, image_width, image_height)

    return distance > (0.2 * image_width)


def closed_fist(hand_landmarks, image_width, image_height, threshold=50):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    finger_tips = [
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP],
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP],
    ]

    distances = [calculate_distance(
        thumb_tip, finger_tip, image_width, image_height) for finger_tip in finger_tips]
    avg_distance = sum(distances) / len(distances)

    return avg_distance < threshold


def middle_finger(hand_landmarks):
    middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]

    index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]

    ring_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    ring_finger_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP]

    pinky_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
    pinky_finger_pip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP]

    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

    middle_raised = middle_finger_tip.y < middle_finger_pip.y

    middle_highest = middle_finger_tip.y < thumb_tip.y

    fingers_closed = (
        index_finger_tip.y > index_finger_pip.y and
        ring_finger_tip.y > ring_finger_pip.y and
        pinky_finger_tip.y > pinky_finger_pip.y
    )

    return middle_raised and middle_highest and fingers_closed
