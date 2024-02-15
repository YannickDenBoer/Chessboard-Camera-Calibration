import cv2 as cv
import numpy as np
import os

class CameraCalibration():
    def __init__(self, path, chessboard):
        self.path = path
        self.chessboard = chessboard
        self.boardsize = chessboard.boardsize
        self.corners = []
        self.gray = []

    def cornerFinder(self, image):
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.gray = image_gray
        detected, self.corners = cv.findChessboardCorners(image_gray, chessboard.boardsize, None) #corners from bottom-left to top-right
        if detected:
            cv.drawChessboardCorners(image, chessboard.boardsize, self.corners,True)
        else: # manual calibration
            clicked_corners = []
            cv.putText(image,"No chessboard detected, please left-click the four corners",(0,30), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2, cv.LINE_AA) #TODO: placeholder
            cv.imshow('Image', image)
            cv.setMouseCallback('Image', self.leftClick, [image, clicked_corners])
            cv.waitKey(0)
            cv.destroyAllWindows
        return self.corners
    
    def leftClick(self, event, x, y, flags, params): 
        # Draw point and save to corner_points
        if len(params[1]) < 4:
            if event == cv.EVENT_LBUTTONDOWN:
                params[1].append([x,y])
                cv.circle(params[0], (x,y), 5, (0,0,255), -1)
                cv.imshow("Image", params[0])

                # Attempt at drawing chessboard points #TODO: does not currently take into account depth of chessboard
                if len(params[1]) == 4:
                    self.corners = np.array(self.findImgPoints(params[1])).astype('float32')
                    cv.drawChessboardCorners(params[0], chessboard.boardsize, self.corners,True)
                    cv.imshow("Image", params[0])
                    cv.waitKey(0)
                    cv.destroyAllWindows()

    def findImgPoints(self, cornerpoints):
        imgpoints = []
        for i in range(self.boardsize[1]):
            for j in range(self.boardsize[0]-1, -1, -1):
                # order: bottom left, top left, bottom right, top right
                a,b,c,d = np.array(cornerpoints[0:4])
                p = a+i/(self.boardsize[1]-1) * (b-a)
                q = c+i/(self.boardsize[1]-1) * (d-c)
                point = p+j/(self.boardsize[0]-1) *(q-p) #/7
                imgpoints.append([point])
        return imgpoints
    
    # takes a range of images (first to last) and the chessboard, and finds the camera calibration info
    def calibrate(self):
        allobjpoints = []
        allimgpoints = []
        for filename in os.listdir(os.path.join(os.getcwd(),self.path)):
            f = os.path.join(path,filename)
            print(f)
            self.corners = []
            img = cv.imread(f)
            self.cornerFinder(img)
            if False: #Put True for troubleshooting
                cv.imshow("Image", img)
                cv.waitKey(0)
            allobjpoints.append(chessboard.objpoints)
            allimgpoints.append(self.corners)
        _, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(allobjpoints, allimgpoints, self.gray.shape[::-1], None, None)
        return cameraMatrix, dist, rvecs, tvecs

class Chessboard():
    def __init__(self, boardsize, squarelength):
        self.boardsize = boardsize
        self.squarelength = squarelength

        objp = np.zeros((boardsize[1]*boardsize[0],3), np.float32)
        objp[:,:2] = np.mgrid[0:boardsize[0],0:boardsize[1]].T.reshape(-1,2)
        #objp *= squarelength
        self.objpoints = objp

def drawAxes(img, corners, imgpts):
    def tupleofInts(arr):
        return tuple(int(x) for x in arr)
    corner = tupleofInts(corners[0].ravel())
    img = cv.line(img,corner,tupleofInts(imgpts[0].ravel()),(255,0,0),2)
    img = cv.line(img,corner,tupleofInts(imgpts[1].ravel()),(0,255,0),2)
    img = cv.line(img,corner,tupleofInts(imgpts[2].ravel()),(0,0,255),2)
    return img

#calibration
path = "Images"
chessboard = Chessboard((9,6),1)
cc = CameraCalibration(path,chessboard)
cameraMatrix, dist, rvecs, tvecs = cc.calibrate()
print(cameraMatrix)

#testing phase
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
testimg = cv.imread("Images/23.jpg")
cc = CameraCalibration("",chessboard)
corners = cc.cornerFinder(testimg)

_, rvecs, tvecs = cv.solvePnP(chessboard.objpoints, corners, cameraMatrix, dist)
imgpoints, _ = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, dist)

testimg = drawAxes(testimg,corners,imgpoints)
cv.imshow("image:",testimg)
cv.waitKey(0)
cv.destroyAllWindows()