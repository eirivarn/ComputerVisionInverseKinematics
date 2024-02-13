import cv2
import numpy as np


def detect_largest_moving_contour():
    video = cv2.VideoCapture(0)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    prev_largest_contour = None
    prev_center = None

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        _, thresh = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)

        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:

            if prev_largest_contour is not None:
                distances = [cv2.pointPolygonTest(
                    contour, prev_center, True) for contour in contours]
                closest_contour_index = np.argmax(distances)
                largest_contour = contours[closest_contour_index]
            else:
                largest_contour = max(contours, key=cv2.contourArea)

            cv2.drawContours(frame, [largest_contour], -1, (50, 255, 0), 3)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

                # Update previous largest contour, center, and radius
                prev_largest_contour = largest_contour
                prev_center = (cX, cY)
                prev_radius = cv2.contourArea(largest_contour)**0.5

        cv2.imshow("Largest moving contour", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


detect_largest_moving_contour()
