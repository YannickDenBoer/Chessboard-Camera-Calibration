import cv2 as cv
import numpy as np

class ChessboardFinder():
    def __init__(self, file, chessboard):
        self.image = file
        self.grayscale = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        _,self.binary  = cv.threshold(self.grayscale, 127, 255, cv.THRESH_BINARY)
        self.boardsize = chessboard.boardsize
        self.detected = False
        self.imgpoints = []
        self.clicked_corners = []

    def cornerFinder(self): # Find and draw corners automatically
        self.detected, self.imgpoints = cv.findChessboardCorners(self.grayscale, self.boardsize, None) #corners from bottom-left to top-right
        if self.detected:
            cv.drawChessboardCorners(self.image, self.boardsize, self.imgpoints, self.detected)
            for imgp in self.imgpoints:
                print(imgp)
            #TODO: point refinement

    def showImage(self):
        cv.imshow("Image", self.image)
        if not self.detected: # manual calibration
            cv.putText(self.image,"No chessboard detected, please left-click the four corners",(0,30),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2,cv.LINE_AA) #placeholder
            cv.imshow('Image',self.image)
            cv.setMouseCallback('Image',self.leftClick, self.image)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def leftClick(self, event, x, y, flags, params): 
        # Draw point and save to corner_points
        if len(self.clicked_corners) <= 3:
            if event == cv.EVENT_LBUTTONDOWN:
                self.clicked_corners.append([x,y])
                cv.circle(params,(x,y),5, (0,0,255), -1)
                cv.imshow('Image', params)

                # Attempt at drawing chessboard points #TODO: does not currently take into account depth of chessboard
                if len(self.clicked_corners) == 4:
                    self.imgpoints = np.array(self.findImgPoints(self.clicked_corners)).astype('float32')
                    cv.drawChessboardCorners(params, self.boardsize,self.imgpoints,True)
                    cv.imshow("Image", params)
                    cv.waitKey(500)
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

class Chessboard():
    def __init__(self, boardsize, squarelength):
        self.boardsize = boardsize
        self.squarelength = squarelength

        objp = np.zeros((boardsize[0]*boardsize[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:boardsize[1],0:boardsize[0]].T.reshape(-1,2)
        self.objpoints = objp

# takes a range of images (first to last) and the chessboard, and finds the camera calibration info
def calibrate(first, last, chessboard):
    allobjpoints = []
    allimgpoints = []
    for i in range(first, last+1):
        filename = f"Images/{i}.jpg"
        img = cv.imread(filename)
        cbf = ChessboardFinder(img, chessboard)
        # if image should be automatically calibrated, use cornerFinder()
        #cbf.cornerFinder()
        cbf.showImage()
        allobjpoints.append(chessboard.objpoints)
        allimgpoints.append(cbf.imgpoints)

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(allobjpoints, allimgpoints, cbf.grayscale.shape[::-1], None, None)
    print(cameraMatrix)
    #cameraPosition = np.matmul(np.array[rvecs,tvecs],)

chessboard = Chessboard((9,6),1)
calibrate(1, 2, chessboard)
