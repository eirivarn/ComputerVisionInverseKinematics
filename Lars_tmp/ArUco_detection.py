import cv2 as cv
import numpy as np

class ArUcoDetection:
    def __init__(self, name):
        self.img = None
        self.imgName = name
        self.centers = []
        self.allCorners = None
        self.ids = None
        self.arucoSizeRL = 1.7 #cm
        

    def generateArUcoPNG(self, markerID):
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
        marker_size = 200
        markerImage = cv.aruco.generateImageMarker(dictionary, markerID, marker_size)
        filename = f"aruco_marker_{markerID}.png"
        cv.imwrite(filename, markerImage)

        print(f"ArUco marker with ID {markerID} has been generated and saved as '{filename}'")

    def detectMarker(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        parameters = cv.aruco.DetectorParameters()
        arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
        detector = cv.aruco.ArucoDetector(arucoDict, parameters)

        self.allCorners, self.ids, rejectedImgPoints = detector.detectMarkers(gray)
        self.calcCenters()

        if len(self.allCorners) > 0:
            cv.aruco.drawDetectedMarkers(self.img, self.allCorners, self.ids)
        else:
            print("No ArUco markers detected")


    def findDistOfMarkers(self):
        imgSize = self.getImgSize()
        scalar = self.calcScalar(self.allCorners, imgSize)
        print(self.centers)
        xy_dist = np.absolute(self.centers[0]- self.centers[1])
        dist = round(np.sqrt(xy_dist[0]**2+xy_dist[1]**2), 2)
        print(dist)
        realDist = round(dist*self.arucoSizeRL/scalar, 2)
        self.img = cv.line(self.img, self.centers[0], self.centers[1], color=(255, 0, 0), thickness=2)
        centerLine =  self.centers[0] + (self.centers[1]-self.centers[0])//2
        self.img = cv.putText(self.img, f'{realDist}cm', centerLine, cv.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 1, cv.LINE_AA)
        self.plotRes()

    def calcCenters(self):
        for corners in self.allCorners:
            center = self.calcCenter(corners)
            self.centers.append(center)
            self.img = cv.circle(self.img, center, radius=5, color=(0, 0, 255), thickness=-5) #plot the centers

    def calcScalar(self, allCorners, imgSize):
        sizes = []
        for corners in allCorners:
            corners = corners[0]
            size_i = corners[2]-corners[0]
            sizes.append(size_i)
        avgX = np.average(np.array(sizes).T[0])#/imgSize[1]
        avgY = np.average(np.array(sizes).T[1])#/imgSize[0]
        return np.average([avgX, avgY])
    
    def getImgSize(self):
        return self.img.shape

    def calcCenter(self, corners):
            c = np.array(corners).T
            x = int(np.sum(c[0])//4)
            y = int(np.sum(c[1])//4)
            return np.array([x,y])

    def plotRes(self):
        cv.imshow(self.imgName, self.img)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()
    
    def readImage(self, filename):
        self.img = cv.imread(filename)

def main():
    detector = ArUcoDetection('open1')

    filename = 'Lars_tmp/images/open1ArUco.png'
    detector.readImage(filename)

    detector.detectMarker()
    detector.findDistOfMarkers()

main()