import cv2 as cv
import numpy as np
from manim import *
from manim.utils.file_ops import open_file as open_media_file
import os
import pickle

# Obtain the camera position in world space
def getWorldCameraPos(objpoints, imgpoints, gray):
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera([objpoints], [imgpoints], gray.shape[::-1], None, None)
    _,rvecs, tvecs = cv.solvePnP(objpoints, imgpoints, cameraMatrix, dist)
    rotM = cv.Rodrigues(rvecs)[0]
    initCamPos = -np.matrix(rotM).T * np.matrix(tvecs)
    return [number for matrix in initCamPos for number in matrix.flat]

# Create 3D scene and plot the chessboard corners and camera
class Plot3D(ThreeDScene):
    def construct(self):
        for i, camPos in enumerate(allcamerapos):
            camLabel = Text('cam' + str(i+1) + ': (' + str(round(camPos[0],2)) + ', ' + str(round(camPos[1], 2)) + ', ' + str(round(camPos[2], 2)) + ')').scale(0.15)
            #cameraLabel.to_corner(DR)
            camPoint = Dot3D(camPos, radius = 0.1, color=BLUE)
            camLabel.next_to(camPoint, UP)
            self.add(camPoint)
            self.add_fixed_orientation_mobjects(camLabel)

<<<<<<< HEAD
        remapped_pts = []
        for co in self.chessboard.objpoints:
            remapped_pts.append([co[0], co[2], co[1]])
            self.add(Dot3D(remapped_pts[-1], radius= 0.05, color=RED))

        self.add(Arrow3D(remapped_cp, remapped_pts[0], thickness=0.03, color=YELLOW))
        self.add(Arrow3D(remapped_cp, remapped_pts[8], thickness=0.03, color=GREEN))
        self.add(Arrow3D(remapped_cp, remapped_pts[53], thickness=0.03, color=YELLOW))
        self.add(Arrow3D(remapped_cp, remapped_pts[45], thickness=0.03, color=GREEN))
=======
        self.set_camera_orientation(phi=280 * DEGREES, theta=-45 * DEGREES, zoom=0.25, frame_center=[3,0,-8])
        axes = ThreeDAxes(x_range=[-20,20,5], y_range=[-20,20,5], z_range=[-20,20,5], x_length=40, y_length=40, z_length=40, axis_config={"include_numbers": True})
        labels = axes.get_axis_labels(Tex("x").scale(2), Text("y").scale(2), Text("z").scale(2))
        self.add(axes, labels)
        for co in chessboard.objpoints:
            self.add(Dot3D(co, radius= 0.05, color=RED))
>>>>>>> f986f7089af3006de0869a5c997c3b47c3a03beb


class CameraCalibration():
    def __init__(self, path, chessboard):
        self.path = path
        self.chessboard = chessboard
        self.boardsize = chessboard.boardsize
        self.corners = []
        self.gray = []
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def cornerFinder(self, image, isvideo=False, scaling=1):
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.gray = image_gray
        detected, corners1 = cv.findChessboardCorners(image_gray, self.chessboard.boardsize, None) #corners from bottom-left to top-right
        if not isvideo:
            if detected:
                self.corners = cv.cornerSubPix(image_gray, corners1, (5,5), (-1,-1), self.criteria)
                cv.drawChessboardCorners(image, self.chessboard.boardsize, self.corners,True)
            else: # manual calibration
                clicked_corners = []
                cv.putText(image,"No chessboard detected, please left-click the four corners",(0,30), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2, cv.LINE_AA) #TODO: placeholder
                cv.imshow('Image', image)
                cv.setMouseCallback('Image', self.leftClick, [image, clicked_corners])
                cv.waitKey(0)
                cv.destroyAllWindows
        else:
            self.corners = corners1
        return self.corners, detected
    
    def leftClick(self, event, x, y, flags, params): 
        # Draw point and save to corner_points
        if len(params[1]) < 4:
            if event == cv.EVENT_LBUTTONDOWN:
                params[1].append([x,y])
                cv.circle(params[0], (x,y), 5, (0,0,255), -1)
                cv.imshow("Image", params[0])

                if len(params[1]) == 4:
                    # Calibrate Manually from topleft -> topright -> bottomleft -> bottomright
                    objcorners = np.array([[self.chessboard.boardsize[0]-1,self.chessboard.boardsize[1]-1], [0, self.chessboard.boardsize[1]-1], [self.chessboard.boardsize[0]-1,0], [0,0]],dtype=np.float32) #bl, tl, tr, br-> tl, tr, bl, br
                    objpoints2d = np.array([self.chessboard.objpoints[:,:2]]) # Drop z-axis
                    destpts = np.array(params[1],dtype=np.float32) # Target points

                    # Find homography matrix between obj cornerpoints and img cornerpoints
                    mat = cv.getPerspectiveTransform(objcorners,destpts)

                    # Transform objpoints to imagepoints
                    corners1 = cv.perspectiveTransform(objpoints2d,mat)
                    self.corners = cv.cornerSubPix(self.gray, corners1, (5,5), (-1,-1), self.criteria)

                    cv.drawChessboardCorners(params[0], self.chessboard.boardsize, self.corners,True)
                    cv.imshow("Image", params[0])
                    cv.waitKey(0)
                    cv.destroyAllWindows()
    
    # takes images from path and takes the chessboard, and returns the camera calibration info
    def calibrate(self):
        allobjpoints = []
        allimgpoints = []
        global allcamerapos
        allcamerapos = []
        for filename in os.listdir(os.path.join(os.getcwd(),self.path)):
            f = os.path.join(self.path,filename)
            print(f)
            self.corners = []
            img = cv.imread(f)
            self.cornerFinder(img)
            
            # True shows the cornerpoints of each image
            if False: 
                cv.imshow("Image", img)
                cv.waitKey(0)

            allobjpoints.append(self.chessboard.objpoints)
            allimgpoints.append(self.corners)
<<<<<<< HEAD
            #run3dplot(self.chessboard.objpoints, self.corners, self.gray)
=======
            allcamerapos.append(getWorldCameraPos(chessboard.objpoints, self.corners, self.gray))
>>>>>>> f986f7089af3006de0869a5c997c3b47c3a03beb

        _, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(allobjpoints, allimgpoints, self.gray.shape[::-1], None, None)
        return cameraMatrix, dist, rvecs, tvecs

#Initialise a chessboard
class Chessboard():
    def __init__(self, boardsize, squarelength):
        self.boardsize = boardsize
        self.squarelength = squarelength

        objp = np.zeros((boardsize[1]*boardsize[0],3), np.float32)
        objp[:,:2] = np.mgrid[0:boardsize[0],0:boardsize[1]].T.reshape(-1,2)
        self.objpoints = objp

# Class with operations for drawing Box or Axes
class D3:
    def point(arr):
        return tuple(int(x) for x in arr.squeeze())

    def drawAxes(img, imgpts):
        corner = D3.point(imgpts[0])
        img = cv.line(img,corner,D3.point(imgpts[1]),(255,0,0),2) #X-axis (Blue)
        img = cv.line(img,corner,D3.point(imgpts[2]),(0,255,0),2) #Y-axis (Green)
        img = cv.line(img,corner,D3.point(imgpts[3]),(0,0,255),2) #Z-axis (Red)
        return img
    
    def drawBox(img, imgpts):
        for i in range(4):
            img = cv.line(img,D3.point(imgpts[i]),D3.point(imgpts[i+4]),(0,255,255),1)
            for j in range(i,4):
                if not ((i==0 and j==3) or (i==1 and j==2)):
                    img = cv.line(img,D3.point(imgpts[i]),D3.point(imgpts[j]),(0,255,255),1)
                    img = cv.line(img,D3.point(imgpts[i+4]),D3.point(imgpts[j+4]),(0,255,255),1)
        return img

# Define axis from origin
axis = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,-3]])
# Define box from origin
box = np.float32([[0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,-2],[2,0,-2],[0,2,-2],[2,2,-2]])

# Real-Time performance using webcam, exit by pressing 'Q'
def video(cc, cameraMatrix, dist, factor=1):
    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        scaled_frame = cv.resize(frame,None,fx=factor, fy=factor,interpolation=cv.INTER_LINEAR)
        corners, detected = cc.cornerFinder(scaled_frame, isvideo=True, scaling=factor)
        
        # If corners are detected in this frame, draw corners and box
        if detected:
            scaled_corners = np.array(corners/factor,dtype=np.float32)
            cv.drawChessboardCorners(frame,cc.chessboard.boardsize, scaled_corners,True)
            _, rvecs, tvecs = cv.solvePnP(cc.chessboard.objpoints, scaled_corners, cameraMatrix, dist) 
            imgpoints, _ = cv.projectPoints(box, rvecs, tvecs, cameraMatrix, dist)
            frame = D3.drawBox(frame,imgpoints) 

        # Show image, exit by pressing 'Q'
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

<<<<<<< HEAD
def main(img_name, calibrate=False, live=False):
    '''
    - img_name:  Test image filename
    - calibrate: Reruns calibration
    - live:      Enables real-time cube detection after calibration
    Return: Camera Intrinsics, test image with 3D axes representation
    '''
    path = "Images"
    chessboard = Chessboard((9,6),1)

    # True reruns the calibration and saves new intrinsic values
    if calibrate: 
        cc = CameraCalibration(path,chessboard)
        cameraMatrix, dist, rvecs, tvecs = cc.calibrate()
        print(cameraMatrix)
        with open('intrinsics.pckl', 'wb') as f: #Store intrinsic camera values
            pickle.dump([cameraMatrix, dist, rvecs, tvecs], f)
        
    # Load intrinsic camera values
    with open('intrinsics.pckl', 'rb') as f: 
        cameraMatrix, dist, rvecs, tvecs = pickle.load(f)
    print('pickl')
    print(rvecs)
    print(tvecs)

    # Test calibration
    testimg = cv.imread(img_name)
    cc = CameraCalibration("",chessboard)
    corners, _ = cc.cornerFinder(testimg)
=======
# Initialisation:
path = "Images/run2"
chessboard = Chessboard((9,6),1)

# ====== Calibration ======

# True reruns the calibration and saves new intrinsic values
if True: 
    cc = CameraCalibration(path,chessboard)
    cameraMatrix, dist, rvecs, tvecs = cc.calibrate()
    print(cameraMatrix)
    with open('intrinsics.pckl', 'wb') as f: #Store intrinsic camera values
        pickle.dump([cameraMatrix, dist, rvecs, tvecs], f)

    # plot all camera world points
    scene = Plot3D()
    scene.render()
    open_media_file(scene.renderer.file_writer.image_file_path)

# ====== Testing ======
# Load intrinsic camera values
with open('intrinsics.pckl', 'rb') as f:
    cameraMatrix, dist, rvecs,tvecs = pickle.load(f)

testimg = cv.imread("Images/test.jpg")
>>>>>>> f986f7089af3006de0869a5c997c3b47c3a03beb

    # Find extrinsics to draw Box and Axes
    _, rvecs, tvecs = cv.solvePnP(cc.chessboard.objpoints, corners, cameraMatrix, dist)
    print('solvepnp')
    print(rvecs)
    print(tvecs)
    # Draw Box
    imgpoints, _ = cv.projectPoints(box, rvecs, tvecs, cameraMatrix, dist)
    testimg = D3.drawBox(testimg,imgpoints)
    # Draw Axes
    imgpoints, _ = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, dist)
    testimg = D3.drawAxes(testimg,imgpoints)

    # Show Test Image
    cv.imshow("image:",testimg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Real-Time Performance
    if live:
        # Factor=1: no rescaling. Factor<1: increases framerate but decreases accuracy
        video(cc, cameraMatrix, dist, factor=1)

if __name__ == "__main__":
    main("Images/5.jpg", calibrate=False)
