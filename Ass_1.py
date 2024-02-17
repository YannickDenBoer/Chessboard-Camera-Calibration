import cv2 as cv
import numpy as np
from manim import *
from manim.utils.file_ops import open_file as open_media_file
import os
import pickle

# obtain the camera position in world space and render manim scene
def run3dplot(objpoints, imgpoints, gray):
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera([objpoints], [imgpoints], gray.shape[::-1], None, None)
    _,rvecs, tvecs = cv.solvePnP(objpoints, imgpoints, cameraMatrix, dist)
    rotM = cv.Rodrigues(rvecs)[0]
    initCamPos = -np.matrix(rotM).T * np.matrix(tvecs)
    global camPos
    camPos = [number for matrix in initCamPos for number in matrix.flat]
    scene = Plot3D()
    scene.render()
    open_media_file(scene.renderer.file_writer.image_file_path)

class Plot3D(ThreeDScene):
    def construct(self):
        remapped_cp = [camPos[0], camPos[2], camPos[1]]
        cameraLabel = Text('(' + str(round(camPos[0],2)) + ', ' + str(round(camPos[1], 2)) + ', ' + str(round(camPos[2], 2)) + ')').scale(0.5)
        self.add_fixed_in_frame_mobjects(cameraLabel)
        cameraLabel.to_corner(DR)
        self.set_camera_orientation(phi=80 * DEGREES, theta=-225 * DEGREES, zoom=0.4, frame_center=[0,0,remapped_cp[2] / 4])
        axes = ThreeDAxes(x_length=40, y_length=40, z_length=40)
        camPoint = Dot3D(remapped_cp, radius = 0.2, color=BLUE)
        labels = axes.get_axis_labels(Tex("x").scale(2), Text("z").scale(2), Text("y").scale(2))
        self.add(axes, camPoint, labels)

        remapped_pts = []
        for co in chessboard.objpoints:
            remapped_pts.append([co[0], co[2], co[1]])
            self.add(Dot3D(remapped_pts[-1], radius= 0.05, color=RED))

        self.add(Arrow3D(remapped_cp, remapped_pts[0], thickness=0.03, color=YELLOW))
        self.add(Arrow3D(remapped_cp, remapped_pts[8], thickness=0.03, color=GREEN))
        self.add(Arrow3D(remapped_cp, remapped_pts[53], thickness=0.03, color=YELLOW))
        self.add(Arrow3D(remapped_cp, remapped_pts[45], thickness=0.03, color=GREEN))

class CameraCalibration():
    def __init__(self, path, chessboard):
        self.path = path
        self.chessboard = chessboard
        self.boardsize = chessboard.boardsize
        self.corners = []
        self.gray = []

    def cornerFinder(self, image, isvideo=False, scaling=1):
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.gray = image_gray
        detected, self.corners = cv.findChessboardCorners(image_gray, chessboard.boardsize, None) #corners from bottom-left to top-right
        if not isvideo:
            if detected:
                cv.drawChessboardCorners(image, chessboard.boardsize, self.corners,True)
            else: # manual calibration
                clicked_corners = []
                cv.putText(image,"No chessboard detected, please left-click the four corners",(0,30), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2, cv.LINE_AA) #TODO: placeholder
                cv.imshow('Image', image)
                cv.setMouseCallback('Image', self.leftClick, [image, clicked_corners])
                cv.waitKey(0)
                cv.destroyAllWindows
        return self.corners, detected
    
    def leftClick(self, event, x, y, flags, params): 
        # Draw point and save to corner_points
        if len(params[1]) < 4:
            if event == cv.EVENT_LBUTTONDOWN:
                params[1].append([x,y])
                cv.circle(params[0], (x,y), 5, (0,0,255), -1)
                cv.imshow("Image", params[0])

                # Attempt at drawing chessboard points #TODO: does not currently take into account depth of chessboard
                if len(params[1]) == 4:
                    #self.corners = np.array(self.findImgPoints(params[1])).astype('float32')
                    
                    objcorners = np.array([[chessboard.boardsize[0]-1,0], [chessboard.boardsize[0]-1,chessboard.boardsize[1]-1], [0,0], [0, chessboard.boardsize[1]-1]],dtype=np.float32) #br, tl, bl, tr
                    objpoints2d = np.array([chessboard.objpoints[:,:2]])

                    destpts = np.array(params[1],dtype=np.float32)
                    mat = cv.getPerspectiveTransform(objcorners,destpts)
                    self.corners = cv.perspectiveTransform(objpoints2d,mat)
                    
                    cv.drawChessboardCorners(params[0], chessboard.boardsize, self.corners,True)
                    cv.imshow("Image", params[0])
                    cv.waitKey(0)
                    cv.destroyAllWindows()

    def findImgPoints(self, cornerpoints):
        imgpoints = []
        for i in range(self.boardsize[1]):
            for j in range(self.boardsize[0]-1, -1, -1):
                # order: bottom left, top left, bottom right, top right
                a,b,c,d = np.array(cornerpoints[0:4]) #bl, tl, br, tr
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
            # run3dplot(chessboard.objpoints, self.corners, self.gray)

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

def video(cc, cameraMatrix, dist, factor=1):
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        scaled_frame = cv.resize(frame,None,fx=factor, fy=factor,interpolation=cv.INTER_LINEAR)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, detected = cc.cornerFinder(scaled_frame, isvideo=True, scaling=factor)

        if detected:
            scaled_corners = corners/factor
            cv.drawChessboardCorners(frame,cc.chessboard.boardsize, scaled_corners,True)
            _, rvecs, tvecs = cv.solvePnP(cc.chessboard.objpoints, scaled_corners, cameraMatrix, dist) 
            imgpoints, _ = cv.projectPoints(box, rvecs, tvecs, cameraMatrix, dist)
            frame = D3.drawBox(frame,imgpoints) 

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

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

#testimg = cv.imread("Images/2.jpg")
testimg = cv.imread("Chessboardtest.jpg")

#Find corners and rotation/translation vectors
cc = CameraCalibration("",chessboard)
corners, _ = cc.cornerFinder(testimg)

if True:
    _, rvecs, tvecs = cv.solvePnP(cc.chessboard.objpoints, corners, cameraMatrix, dist)

    imgpoints, _ = cv.projectPoints(box, rvecs, tvecs, cameraMatrix, dist)
    testimg = D3.drawBox(testimg,imgpoints)

    imgpoints, _ = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, dist)
    testimg = D3.drawAxes(testimg,imgpoints)
    cv.imshow("image:",testimg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    video(cc, cameraMatrix, dist, factor=1) #factor=1: no rescaling. factor<1: speeds up video but decreases accuracy

#TODO: Improve accuracy (pixel optimization), Choice tasks, Add Uncalibrated Images
