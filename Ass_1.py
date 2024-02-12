import cv2 as cv
import numpy as np

filename = "Chessboardtest.jpg"
img = cv.imread(filename)
#grayscale = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#_, binary = cv.threshold(grayscale,127,255,cv.THRESH_BINARY)

class ChessboardFinder():
    def __init__(self, file):
        self.image = file
        self.grayscale = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        _,self.binary  = cv.threshold(self.grayscale, 127, 255, cv.THRESH_BINARY)
        self.boardsize = (7,7)
        self.detected = False
        self.corners = []

    def cornerFinder(self): # Find and draw corners automatically
        self.detected, self.corners = cv.findChessboardCorners(self.grayscale, self.boardsize, None) #corners from bottom-left to top-right
        if self.detected: 
            cv.drawChessboardCorners(self.image, (7,7), self.corners, self.detected)
            #print(self.corners)
            #TODO: point refinement

    def showImage(self):
        cv.imshow("Image", self.image)
        if not self.detected: # manual calibration
            cv.putText(self.image,"No chessboard detected, please left-click the four corners",(0,30),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2,cv.LINE_AA) #placeholder
            cv.imshow('Image',self.image)
            cv.setMouseCallback('Image',leftClick, self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

corner_points = []
def leftClick(event, x, y, flags, params): 
    # Draw point and save to corner_points
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,' ',y)
        corner_points.append([x,y])
        cv.circle(params,(x,y),5, (0,0,255), -1)
        cv.imshow('Image', params)

        # Attempt at drawing chessboard points, incorrect: does not take into account depth of chessboard
        # Reproduce Array with 2d points of findchessboardcorners and input to drawchessboardcorners?
        if len(corner_points) == 4:
            points = findImgPoints(corner_points)
            for i in points:
                sq = np.squeeze(i)
                point = (int(sq[0]), int(sq[1]))
                print(point)
                cv.circle(params, point, 3, (0,0,255), 1)
                cv.imshow('Image', params)          
    return

#incorrect: does not take into account depth of chessboard
def findImgPoints(cornerpoints):
    imgpoints = []
    for i in range(7):
        for j in range(7):
            a,b,c,d = np.array(cornerpoints[0:4])
            p = a+i/6 * (b-a)
            q = c+i/6* (d-c)
            point = p+j/6*(q-p) #/7
            imgpoints.append(point)
    print(np.array(imgpoints))
    return imgpoints

class Chessboard():
    def __init__(self, boardsize, squarelength):
        self.boardsize = boardsize
        self.squarelength = squarelength

        objp = np.zeros((7*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
        self.objpoints = objp

# Read image
# Find imgpoints (corners)
# Find objpoints (chessboardsize, squarelength)
# Pass imgpoints & objpoints to calibration function
cbf = ChessboardFinder(img)
cbf.cornerFinder()
imgpoints = cbf.corners
chessboard = Chessboard((7,7),1)
print(np.array(chessboard.objpoints))
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera([chessboard.objpoints], [imgpoints], cbf.grayscale.shape[::-1], None, None)
print(mtx)