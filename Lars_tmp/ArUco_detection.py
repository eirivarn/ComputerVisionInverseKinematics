import cv2 as cv
import numpy as np

def generateArUcoPNG(markerID):
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    marker_size = 200
    markerImage = cv.aruco.generateImageMarker(dictionary, markerID, marker_size)
    filename = f"aruco_marker_{markerID}.png"
    cv.imwrite(filename, markerImage)

    print(f"ArUco marker with ID {markerID} has been generated and saved as '{filename}'")

def detectMarker(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    parameters = cv.aruco.DetectorParameters()
    arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, arucoDict, parameters=parameters)

    if len(corners) > 0:
        for marker in (corners):
            print(marker)
        cv.aruco.drawDetectedMarkers(img, corners, ids)
        
        #Disp res:
        cv.imshow('Detected ArUco markers', img)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()
        return (corners, ids)
    else:
        print("No ArUco markers detected")

def findDistOfMarkers(allCorners, ids, img):
    centers = []
    for corners in allCorners:
        center = getCenter(corners)
        print(center)
        centers.append(center)
    #     img = cv.circle(img, center, radius=1, color=(0, 0, 255), thickness=-1)
    # cv.imshow('Centers', img)
    # if cv.waitKey(0) & 0xff == 27:
    #         cv.destroyAllWindows()

    # imgSize = getImgSize(img)
    scalar = calcScalar(allCorners)#, imgSize)

def calcScalar(allCorners):#, imgSize):
    for corners in allCorners:
        ysize = corners[0]

def getCenter(corners):
        c = np.array(corners).T
        x = int(np.sum(c[0])//4)
        y = int(np.sum(c[1])//4)
        return (x,y)

def main():
    filename = 'images/open1ArUco.png'
    img = cv.imread(filename)
    allCorners, ids = detectMarker(img)
    findDistOfMarkers(allCorners, ids, img)

main()