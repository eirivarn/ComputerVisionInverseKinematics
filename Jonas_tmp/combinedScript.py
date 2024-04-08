import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from src.utils.imageProcessing import *

def calcCenter(corners):
        c = np.array(corners).T
        x = int(np.sum(c[0])//4)
        y = int(np.sum(c[1])//4)
        return np.array([x,y])

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
        return np.array([0, 0])
   

    cos_theta2 = (d**2 - a1**2 - a2**2) / (2 * a1 * a2)
    theta2 = np.arccos(cos_theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(a2 *
                                           np.sin(theta2), a1 + a2 * np.cos(theta2))

    return np.array([theta1, theta2])


def draw_robot_arm_on_frame(frame, q, image_width, image_height, end_effector_closed):
    a1, a2 = int(0.25 * image_width), int(0.25 * image_height)  # Scale arm lengths to image size
    
    # Calculate joint positions
    joint1 = (int(image_width / 2 + a1 * np.cos(q[0])), int(image_height / 2 + a1 * np.sin(q[0])))
    joint2 = (int(joint1[0] + a2 * np.cos(q[0] + q[1])), int(joint1[1] + a2 * np.sin(q[0] + q[1])))
    
    # Draw arm segments
    cv2.line(frame, (int(image_width / 2), int(image_height / 2)), joint1, (0, 255, 0), 3)  # Base to joint1
    cv2.line(frame, joint1, joint2, (255, 0, 0), 3)  # joint1 to joint2
    
    # Draw circles at the joints
    cv2.circle(frame, joint1, 5, (0, 0, 255), -1)  # Joint1
    cv2.circle(frame, joint2, 5, (0, 0, 255), -1)  # Joint2


   
    # Display the resulting frame
    cv2.imshow('Robot Arm Simulation', frame)


def main(): 
    
    parameters = cv.aruco.DetectorParameters()
    arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    detector = cv.aruco.ArucoDetector(arucoDict, parameters)

    frameWidth = 450
    frameHeight = 640
    cap = cv.VideoCapture(1)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10,150)

    end_effector_position = None

    figure, ax = plt.subplots()
    plt.ion()

    while cap.isOpened():
         
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(image, cv.COLOR_BGR2GRAY)

        allCorners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        if ids is not None and np.any(ids == 1):
            center = calcCenter(allCorners[np.where(ids == 1)[0][0]])
            image = cv.circle(image, center, radius=5, color=(0, 0, 255), thickness=-1)
            end_effector_position = center

        if len(allCorners) > 0:
            cv.aruco.drawDetectedMarkers(image, allCorners, ids)

        if end_effector_position is not None: 
            q = calculate_inverse_kinematics(end_effector_position, image.shape[1], image.shape[0])  # Adjust based on actual function signature
            draw_robot_arm_on_frame(image, q, image.shape[1], image.shape[0], 0)  # Make sure this function correctly plots on 'ax'

        else: 
            draw_robot_arm_on_frame(image, [0, 0], image.shape[1], image.shape[0], 0)

        #cv2.imshow("Hand Tracking", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff() 

main()



     