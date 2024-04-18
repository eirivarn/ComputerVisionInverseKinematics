
import tkinter as tk
from threading import Thread, Event
import cv2
from tkinter import messagebox
from src.ArUco_detection import ArUcoDetection
from filter_magic import main as filter_main
from src.utils.inverse_kinematics import calculate_inverse_kinematics, draw_robot_arm
from mediapipe_magic import hand_tracking_and_control_robot_combined



class App:
    def __init__(self, window, window_title) -> None:
        self.cap = None
        self.image = None
        self.detectionMethod = None
        self.stopEvent = Event() # Event to signal the thread to stop


        self.window = window
        self.window.title(window_title)
        self.createWidgets(window)

        self.arucoDetector = ArUcoDetection("ArUco Detection")
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        
        # Start video capture in a separate thread
        self.videoThread = Thread(target=self.startVideoStream)
        self.videoThread.daemon = True

    def runGui():
        App(tk.Tk(), "Tkinter and OpenCV")
        tk.mainloop()

    def createWidgets(self, window):
        # Button UI
        self.btn_detect1 = tk.Button(window, text="Google", width=50, command=lambda: self.select_method('google'))
        self.btn_detect1.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_detect2 = tk.Button(window, text="ArUco", width=50, command=lambda: self.select_method('aruco'))
        self.btn_detect2.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_detect3 = tk.Button(window, text="Filter", width=50, command=lambda: self.select_method('filter'))
        self.btn_detect3.pack(anchor=tk.CENTER, expand=True)

        self.btn_quit = tk.Button(window, text="Quit", width=50, command=self.quitApp)
        self.btn_quit.pack(anchor=tk.CENTER, expand=True)

    def select_method(self, method):
        self.detectionMethod = method
        # start video stream
        self.restartVideoStream()

    def restartVideoStream(self):
        self.stopEvent.set() # Signal the current thread to stop
        if self.cap.isOpened():
            self.cap.release() # Release the camera resource
        self.stopEvent.clear() # Clear the stop event for the new thread
        self.videoThread = Thread(target=self.startVideoStream, daemon=True)
        self.videoThread.start()

    def startVideoStream(self):
        #start continuous image capture
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened():
            success, self.image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            display = False
            # Perform wanted detection method:
            if self.detectionMethod == None:
                print("No detection method selected")
            elif self.detectionMethod == 'google':
                self.googleHandRecog()
            elif self.detectionMethod == "aruco":    
                self.arucoHandRecog()
                display = True
            elif self.detectionMethod == "filter":
                self.filterHandRecog()
            
            # Display the resulting frame
            if display:
                cv2.imshow("Hand Tracking", self.image)

            # Stop the video stream on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopEvent.set()
                break

        #Close the window and release the camera resource
        self.cap.release()
        cv2.destroyAllWindows()


    def googleHandRecog(self):
        hand_tracking_and_control_robot_combined()

    def arucoHandRecog(self):
        handCenter, openState = self.arucoDetector.getHandInfo(self.image)
        if handCenter is not None:
            # handCenter = (handCenter[0], self.image.shape[1]/2 - handCenter[1])
            self.drawInversedKinematics(handCenter, openState)

    def filterHandRecog(self):
        filter_main()

    def drawInversedKinematics(self, handCenter, openState):
        if handCenter is not None:
            q = calculate_inverse_kinematics(handCenter, self.image.shape[1], self.image.shape[0])
            if q is not None:
                draw_robot_arm(self.image, q, openState)

        # print('center:', handCenter, 'open:', openState)
        


    def quitApp(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.window.destroy()


if __name__ == "__main__":
    App.runGui()