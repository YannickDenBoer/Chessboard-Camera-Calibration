import cv2 as cv

filename = "Chessboardtest.jpg"
img = cv.imread(filename)
grayscale = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
_, binary = cv.threshold(grayscale,127,255,cv.THRESH_BINARY)

# Find and draw corners automatically
detected, corners = cv.findChessboardCorners(grayscale,(7,7),None)
if detected: 
    cv.drawChessboardCorners(img, (7,7),corners,detected)
    #TODO: point refinement

# TODO: manually calibrate points
# leftclick event called by setMouseCallback
def leftClick(event, x, y, flags, params): 
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,' ',y)
        cv.circle(params,(x,y),5, (0,0,255), -1)
        cv.imshow('Image', params)
    return

def showImage(image):
    cv.imshow("Image", image)
    cv.setMouseCallback('Image',leftClick, image) #Allows for drawing points on image (manual calibration)
    cv.waitKey(0)
    return

showImage(img)