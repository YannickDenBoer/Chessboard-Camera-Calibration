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

    def cornerFinder(self): # Find and draw corners automatically
        self.detected, corners = cv.findChessboardCorners(self.grayscale, self.boardsize, None) #corners from bottom-left to top-right
        if self.detected: 
            cv.drawChessboardCorners(self.image, (7,7), corners, self.detected)
            #TODO: point refinement

    def showImage(self):
        cv.imshow("Image", self.image)
        if not self.detected: # manual calibration
            cv.putText(self.image,"No chessboard detected, please left-click the four corners",(0,30),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2,cv.LINE_AA) #placeholder
            cv.imshow('Image',self.image)
            cv.setMouseCallback('Image',leftClick, self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

# leftclick event called by setMouseCallback
corner_points = []
def leftClick(event, x, y, flags, params): 
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,' ',y)
        corner_points.append([x,y])
        cv.circle(params,(x,y),5, (0,0,255), -1)
        cv.imshow('Image', params)

        if len(corner_points) == 4:
            points = findImgPoints(corner_points)
            #cv.drawChessboardCorners(params,(7,7),np.array(points),False)
            for i in points:
                sq = np.squeeze(i)
                point = (int(sq[0]), int(sq[1]))
                print(point)
                cv.circle(params, point, 3, (0,0,255), 1)
                cv.imshow('Image', params)          
    return

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

#findImgPoints([[372,435], [498,136], [1065,446], [936,142]])
#showImage(img)
cbf = ChessboardFinder(img)
cbf.cornerFinder()
cbf.showImage()