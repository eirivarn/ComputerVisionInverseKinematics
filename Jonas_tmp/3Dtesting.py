import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from src.utils.imageProcessing import *


class ArUcoHandler: 

    def __init__(self) -> None:
        self.parameters = cv.aruco.DetectorParameters()
        self.arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
        self.detector = cv.aruco.ArucoDetector(self.arucoDict, self.parameters)

    def detectMarkers(self, gray) -> any:
        """
        Detects aruco markers in image if any.
        The index of ID from the ids array correspond with the corners belonging to that marker in the allCorners arr. 
        """

        allCorners, ids, rejectedImgPoints = self.detector.detectMarkers(gray) 

        return allCorners, ids

    def findClosestCorners(self, corners:np.array) -> np.array:
        
        """
        Function used to assign plane to the marker.
        Finds the two closest corners to the first corner
        """

        mainCorner = corners[0]
        corners = np.delete(corners, 0,0)

        distances = [np.linalg.norm(mainCorner - corner) for corner in corners]
        corners = np.delete(corners, np.argmax(distances), 0)

        return corners
    
    def estimate_H(self, xy, XY):
    # Tip: U,s,VT = np.linalg.svd(A) computes the SVD of A.
    # The column of V corresponding to the smallest singular value
    # is the last column, as the singular values are automatically
    # ordered by decreasing magnitude. However, note that it returns
    # V transposed.

        n = XY.shape[1]
        A = []
        for i in range(n): 

            ai1 = np.array([XY[0, i], XY[1, i], 1, 0, 0, 0, -XY[0, i]*xy[0, i], -XY[1, i]*xy[0, i], -xy[0, i]])
            ai2 = np.array([0, 0, 0, XY[0, i], XY[1, i], 1, -XY[0, i]*xy[1, i], -XY[1, i]*xy[1, i], -xy[1, i]])
            A.append(ai1)
            A.append(ai2)
            

        A = np.array(A).reshape((2*n,9))

        U, s, VT = np.linalg.svd(A)

        h = VT[-1,:]

        H = h.reshape((3,3))

        return H

class DrawHandler() : 
    def __init__(self) -> None:
        pass

    def drawVector(self, image, startPoint, endPoint): 

        image = cv.arrowedLine(image, startPoint, endPoint, color=(0, 0, 255), thickness=5)

        return image
    
    def draw_vectors_3d(self, origin, directions):
        """
        Draws vectors in 3D space.
        
        Parameters:
        - origins: An Nx3 numpy array of origin points (where N is the number of vectors)
        - directions: An Nx3 numpy array representing the direction and magnitude of each vector from its origin
        
        Each row in `origins` and `directions` corresponds to the origin and direction (dx, dy, dz) of a vector.
        """
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Set the aspect ratio to 'auto' to avoid squishing the plot
        ax.set_aspect('auto')
        
        for direction in directions:
            # Here, `origin` is the starting point of the vector, and `direction` indicates where the vector points and its magnitude.
            ax.quiver(origin, origin, origin, direction[0], direction[1], direction[2],
                    arrow_length_ratio=0.1, color='blue')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


def main() : 
    I = cv.imread("/Users/jonasolsen/Documents/Skole/IIkt/6_semester/TTK4255_Robotsyn/prosjekt/ComputerVisionInverseKinematics/Jonas_tmp/data/raw/img/3dBase3.jpg", cv.COLOR_BGR2RGB)

    arucoHandler = ArUcoHandler()

    corners, ids = arucoHandler.detectMarkers(I)
    
    if corners is not None: 
        cv.aruco.drawDetectedMarkers(I, corners, ids)

        singleMarkerId = ids[0][0]

        singleMarkerCorners = corners[np.where(ids==singleMarkerId)[0][0]][0].reshape((2,4))

        print(singleMarkerCorners.shape)

        objPoints = np.array([[0, 0, 0], [0, -5, 0], [-5, -5, 0], [-5, 0, 0]]).reshape((3, 4))

        H = arucoHandler.estimate_H(singleMarkerCorners, objPoints)

        print(H)


        I = cv.drawMarker(I, singleMarkerCorners[1].astype(int), color=(0, 0, 255), markerSize=5, thickness=10, markerType=1)
        I = cv.drawMarker(I, singleMarkerCorners[2].astype(int), color=(0, 0, 255), markerSize=5, thickness=10, markerType=1)
        I = cv.drawMarker(I, singleMarkerCorners[3].astype(int), color=(0, 0, 255), markerSize=5, thickness=10, markerType=1)

        

        


    cv.imshow("image", I)

    cv.waitKey(0)



    
main()

