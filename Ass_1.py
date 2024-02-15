import cv2 as cv
import numpy as np
import os
import pickle

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
            if False: #True shows the cornerpoints of each image
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
        self.objpoints = objp

class D3:
    def point(arr):
        return tuple(int(x) for x in arr.squeeze())

    def drawAxes(img, imgpts):
        corner = D3.point(imgpts[0])
        img = cv.line(img,corner,D3.point(imgpts[1]),(255,0,0),2)
        img = cv.line(img,corner,D3.point(imgpts[2]),(0,255,0),2)
        img = cv.line(img,corner,D3.point(imgpts[3]),(0,0,255),2)
        return img
    
    def drawBox(img, imgpts):
        for i in range(4):
            img = cv.line(img,D3.point(imgpts[i]),D3.point(imgpts[i+4]),(0,255,255),1)
            for j in range(i,4):
                if not ((i==0 and j==3) or (i==1 and j==2)):
                    img = cv.line(img,D3.point(imgpts[i]),D3.point(imgpts[j]),(0,255,255),1)
                    img = cv.line(img,D3.point(imgpts[i+4]),D3.point(imgpts[j+4]),(0,255,255),1)
        return img

axis = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,-3]])
box = np.float32([[0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,-2],[2,0,-2],[0,2,-2],[2,2,-2]])

#calibration
path = "Images"
chessboard = Chessboard((9,6),1)

if False: #True reruns the calibration and saves new intrinsic values
    cc = CameraCalibration(path,chessboard)
    cameraMatrix, dist, rvecs, tvecs = cc.calibrate()
    print(cameraMatrix)
    with open('intrinsics.pckl', 'wb') as f: #Store intrinsic camera values
        pickle.dump([cameraMatrix, dist, rvecs, tvecs], f)
        
#testing phase
with open('intrinsics.pckl', 'rb') as f: #Load intrinsic camera values
    cameraMatrix, dist, rvecs,tvecs = pickle.load(f)


testimg = cv.imread("Images/2.jpg")

#Finds corners and rotation/translation vectors
cc = CameraCalibration("",chessboard)
corners = cc.cornerFinder(testimg)
_, rvecs, tvecs = cv.solvePnP(chessboard.objpoints, corners, cameraMatrix, dist)

imgpoints, _ = cv.projectPoints(box, rvecs, tvecs, cameraMatrix, dist)
testimg = D3.drawBox(testimg,imgpoints)

imgpoints, _ = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, dist)
testimg = D3.drawAxes(testimg,imgpoints)
cv.imshow("image:",testimg)
cv.waitKey(0)
cv.destroyAllWindows()