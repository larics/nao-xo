'''
NaoXO, v1.0
As of 18.3.2014.

@author: FP
'''

## import ALProxy to connect to Naoqi on the robot
from naoqi import ALProxy

## import numpy for matrix operations and OpenCV
import numpy as np

## import OpenCV for visualization
import cv2

## import definitions for xo class
from naoxo_definitions import *

## import image processing module
import imgproc_xo as img

## import NAO vision definitions for image grabbing
from vision_definitions import kVGA, kBGRColorSpace

## import time module
import time

## import strategy modules
## TODO: rewrite strategy modules
from pobjeda import pobjeda
from krizic_kruzic_strategija import strategija_x
from krizic_kruzic_strategija import strategija_o


class NaoXO():
    '''
    Class used to initialize and play tic-tac-toe with NAO
    '''

    def __init__(self, IP, PORT):
        '''
        Constructor for NAO_XO class, takes IP and PORT of NAOqi running on the robot
        '''
        
        ## proxies to NAOqi modules
        self.motion = ALProxy("ALMotion", IP, PORT)
        self.memory = ALProxy("ALMemory", IP, PORT)
        self.behavior = ALProxy("ALBehaviorManager", IP, PORT)
        self.tts = ALProxy("ALTextToSpeech", IP, PORT)
        self.videoProxy = ALProxy("ALVideoDevice", IP, PORT)
        
        ## set video parameters
        ## select bottom camera
        self.videoProxy.setParam(18,1)
        ## subscribe to ALVideoDevice
        self.video = self.videoProxy.subscribe("video", kVGA, kBGRColorSpace, 30)
        ## create image header
        alimg = self.videoProxy.getImageRemote(self.video)
        self.size = (alimg[0], alimg[1])
        self.imgheader = cv2.cv.CreateImageHeader((alimg[0], alimg[1]), cv2.cv.IPL_DEPTH_8U, 3)
        ## create image processing object
        self.imgproc = img.ImgProcessingXO(self.size)
        print("[INFO ] Image processing initialized")
        ## state of the game
        self.state = [0]*9
        self.board = [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
        self.mode = 'x'
        
        self.cameraInit()
        self.setFieldSize(0.06)
        
        
    def cameraInit(self, params = intrinsic_params):
        '''
        cameraInit method is used to intialize camera matrix with intrinsic camera parameters
        params = [fx, fy, cx, cy, s] with:
            fx = focal lenght in x axis
            fy = focal length in y axis (usually fx=fy)
            cx = image center in x axis
            cy = image center in y axis
            s  = skew factor (usually s=0)
        '''
        
        ## camera intrinsic parameters
        self.camMatrix = np.zeros((3,3),dtype=np.float64)
        
        self.camMatrix[0][0] = params[0] ## fx
        self.camMatrix[1][1] = params[1] ## fy
        self.camMatrix[0][2] = params[2] ## cx
        self.camMatrix[1][2] = params[3] ## cy
        self.camMatrix[0][1] = params[4] ## s
        self.camMatrix[2][2] = 1.0
    
    def setFieldSize(self, dim):
        ''' 
        setFieldSize method is used to set the dimensions of the playing field
        objPoints contains coordinates of the vertices of the inner square on the field in local coordinate frame,
        which is placed in the center of the square.
        fieldPoints contains coordinates of the centers of each box on the playing field, with respect to the local coordinate frame
        z coordinate of fieldPoints is set to zero, can be changed through manipulationInit method
        Corner points mark the corners of the playing field, and are used to evaluate if objects are outside of the playing field
        '''
         
        d = dim / 2.0
        
        ## coordinates of the vertices of inner square of the playing field in local coordinate frame
        self.objPoints = np.zeros((4,3), dtype=np.float64)
        
        self.objPoints[0,0] = -d ## first vertex = top left
        self.objPoints[0,1] =  d
        
        self.objPoints[1,0] = -d ## second vertex = bottom left
        self.objPoints[1,1] = -d
        
        self.objPoints[2,0] =  d ## third vertex = bottom right
        self.objPoints[2,1] = -d
        
        self.objPoints[3,0] =  d ## fourth vertex = top right
        self.objPoints[3,1] =  d
        
        d2 = dim
        
        ## coordinates of the centers of all boxes on the playing field in local coordainte frame
        self.fieldPoints = np.zeros((4,9), dtype=np.float64)
        
        self.fieldPoints[0,0] = -d2     ## first box = top left
        self.fieldPoints[1,0] =  d2
        self.fieldPoints[3,0] =  1
        
        self.fieldPoints[0,1] = 0.0     ## second box = top middle
        self.fieldPoints[1,1] = d2
        self.fieldPoints[3,1] = 1
        
        self.fieldPoints[0,2] = d2      ## third box = top right
        self.fieldPoints[1,2] = d2
        self.fieldPoints[3,2] = 1
        
        self.fieldPoints[0,3] = -d2     ## fourth box = middle left
        self.fieldPoints[1,3] = 0.0
        self.fieldPoints[3,3] = 1
        
        self.fieldPoints[0,4] = 0.0     ## fifth box = middle
        self.fieldPoints[1,4] = 0.0
        self.fieldPoints[3,4] = 1
        
        self.fieldPoints[0,5] = d2      ## sixth box = middle right
        self.fieldPoints[1,5] = 0.0
        self.fieldPoints[3,5] = 1
        
        self.fieldPoints[0,6] = -d2     ## seventh box = bottom left
        self.fieldPoints[1,6] = -d2
        self.fieldPoints[3,6] = 1
        
        self.fieldPoints[0,7] = 0.0     ## eeighth box = bottom middle
        self.fieldPoints[1,7] = -d2
        self.fieldPoints[3,7] = 1
        
        self.fieldPoints[0,8] =  d2     ## ninth box = bottom right
        self.fieldPoints[1,8] = -d2
        self.fieldPoints[3,8] = 1
        
        ## set cornerPoints
        d3 = 1.5 * dim
        self.cornerPoints = np.zeros((4,4), dtype=np.float64)
        
        self.cornerPoints[0,0] = -d3    ## first corner top left
        self.cornerPoints[1,0] = d3
        self.cornerPoints[3,0] = 1
        
        self.cornerPoints[0,1] = -d3    ## second corner bottom left
        self.cornerPoints[1,1] = -d3
        self.cornerPoints[3,1] = 1
        
        self.cornerPoints[0,2] = d3    ## third corner bottom right
        self.cornerPoints[1,2] = -d3
        self.cornerPoints[3,2] = 1
        
        self.cornerPoints[0,3] = d3    ## fourth corner top right
        self.cornerPoints[1,3] = d3
        self.cornerPoints[3,3] = 1
        
    def manipulationInit(self, height, orientation=goal_orientation, rightSafe=right_safe, leftSafe=left_safe):
        ''' 
        manipulationInit method is used to set safe positions for both hands through rightSafe and leftSafe parameters, 
        which are organized as list with 6 entries [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
        parameter height is used to set the clearance of the hand with respect to the playing field when playing, 
        which is done by increasing the local coordinates of goal points
        parameter orientation is a list containing 3 entries and denotes the orientation that robots hand will assume at the goal point
        '''
        
        ## added height for manipulation
        self.height = height
        
        ## goal point orientation
        self.orient = orientation
        
        ## safe positions for hands
        self.rightSafe = rightSafe
        self.leftSafe = leftSafe
        
    def stanceInit(self):
        '''
        Used to make robot stand and position itself correctly
        '''
        
        ## Set stiffnesses, enableing robot to stand up.
        self.motion.setStiffnesses("Body", 1.0)
        self.behavior.runBehavior("Stand up")
        ## Go to walk init pose
        self.motion.walkInit()
        
        ## Release stiffnesses of the arms
        self.motion.setStiffnesses("RArm", 0.0)
        self.motion.setStiffnesses("LArm", 0.0)
        
        ## Release head stiffness
        self.motion.setStiffnesses("Head", 0.0)
        
        ## initialize manipulation
        ## TODO: remove hard coding of the goal position height clearance
        self.manipulationInit(0.05)
        
        print("[INFO ] Robot pose initialized")
        
    def getImage(self):
        '''
        Gets the image from NAO's camera by using getImageRemote method of ALVideoDevice module
        Converts alimage data into OpenCV format for further processing
        '''
        alimg = self.videoProxy.getImageRemote(self.video)
        cv2.cv.SetData(self.imgheader, alimg[6])
        self.img = np.asarray(self.imgheader[:,:])
        return self.img
    
    def findField(self):
        '''
        Uses pattern matching to find the field and solves PnP to obtain the pose of the field in FRAME_ROBOT
        '''

        ## obtain the image
        self.img = self.getImage()
        ## find and merge lines
        self.lines = self.imgproc.mergeEndPoints(self.imgproc.preprocessLines(self.img), 0.05)
        ## find and index intersections
        self.intersections = self.imgproc.getIndexedIntersections(self.lines)
        ## index lines
        self.lines = self.imgproc.indexLines(self.intersections)
        ## field homogeneous transformation matrix
        self.T_field = np.zeros((4,4), dtype=np.float64)
        ## if there are not exactly four intersections return empty matrix
        if not self.intersections:
            return False
        if not len(self.intersections)==4:
            return False
        
        ## otherwise, calculate position by using solvePnP from OpenCV
        ## create image points from intersections
        imgPoints = np.zeros((4,2), dtype=np.float64)
        for i in range(4):
                imgPoints[i,0]=self.intersections[i][0]
                imgPoints[i,1]=self.intersections[i][1]
        ## object points are created by calling setFieldSize method and can be used for solving PnP
        self.rvec, self.tvec = cv2.solvePnP(self.objPoints, imgPoints, self.camMatrix, distCoeffs = None)
        ## rotation matrix
        R = cv2.cv.CreateMat(3,3,cv2.cv.CV_64FC1)
        ## convert rvec to R
        cv2.cv.Rodrigues2(cv2.cv.fromarray(self.rvec), R)
        
        ## compose homogeneous transformation matrix
        
        Rt = np.zeros((3,4), dtype=np.float64)
        T = np.zeros((4,4), dtype = np.float64)
        T[3,3]=1.0
        for i in range(3):
            for j in range(3):
                T[i,j]=R[i,j]
                Rt[i,j]=R[i,j]
            T[i,3]=self.tvec[i]
            Rt[i,3]=self.tvec[i]
            
                
        ## obtain transformation matrix of the camera in FRAME_ROBOT
        T_camAL = self.motion.getTransform("CameraBottom",2,True)
        ## convert to numpy format, needed for matrix multiplication
        T_camAL = np.asarray(T_camAL)
        T_camAL = np.reshape(T_camAL, (4,4))
        
        ## compose transformation from ALMotion coordinate frame to OpenCV coordinate frame
        T_alCV = np.zeros((4,4), dtype = np.float64)
        T_alCV[0,2] =  1    # this means Z'=X
        T_alCV[1,0] = -1    # this means X'=-Y
        T_alCV[2,1] = -1    # this means Y'=-Z
        T_alCV[3,3] =  1    # homogenous coordinates!
        
        ## transformation of OpenCV camera frame in FRAME ROBOT        
        T_camCV = np.dot(T_camAL, T_alCV)
        
        ## transformation of the playing field in FRAME_ROBOT
        self.T_field = np.dot(T_camCV, T)
        
        ## compose projection matrix
        self.P = np.dot(self.camMatrix, Rt)
        
        ## calculate bounds of the playing field
        self.imgproc.getBounds(self.P, self.cornerPoints)        
        
        return True
    
    def cleanup(self):
        '''
        Cleaning up when exiting
        '''
        
        ## unsubscribe from video device module
        self.videoProxy.unsubscribe(self.video)
        
        ## close all OpenCV windows
        cv2.destroyAllWindows()
    
    def drawstuff(self, flag):
        '''
        Used to draw data, useful for debugging and visualization
        '''
        
        ## open new window
        cv2.namedWindow("Image")

        ## if flag is set to false, just show the original image
        ## else draw valid intersections and bound of th eplaying field
        if flag:
            for i in range(4):
                pt1 = (cv2.cv.Round(self.imgproc.corners[0][i]), cv2.cv.Round(self.imgproc.corners[1][i]))
                pt2 = (cv2.cv.Round(self.imgproc.corners[0][(i-1)%4]), cv2.cv.Round(self.imgproc.corners[1][(i-1)%4]))
                cv2.line(self.img, pt1, pt2, (0,255,0), 2, 8)
                cv2.circle(self.img, pt1, 5, (0,0,255), cv2.cv.CV_FILLED)
            for point in self.intersections:
                cv2.circle(self.img, point, 5, (255,0,0), cv2.cv.CV_FILLED)
        
        ## show image
        cv2.imshow("Image", self.img)
        ## imshow does not work without wait key method
        cv2.waitKey(1)
        
    def checkResult(self):
        '''
        Checks if game is over and what is the outcome
        '''
        
	
        if pobjeda(self.board) == 'nerjeseno':
	    ## return that the game is draw
            return 0
        if self.mode == 'x' and pobjeda(self.board) == 'pobjeda x':
	    ## robot won
            return 1
        if self.mode == 'x' and pobjeda(self.board) == 'pobjeda o':
            ## opponent won
            return 2
        if self.mode == 'o' and pobjeda(self.board) == 'pobjeda o':
            ## robot won
            return 1
        if self.mode == 'o' and pobjeda(self.board) == 'pobjeda x':
            ## opponent won
            return 2
        ## default return
        return -1
    
    def checkValidity(self, new_state, old_state, mode):
        ''' 
        Checks validity of the game state
        Counts objects in the old state and the new state. For state to be valid, 
        there must be no differences between old and new state in those boxes on 
        the playing field that were occupied in the old state.
        Additionally, the number of the objects opponent is playing with must increase by one
        '''
        
        ## count objects in the old state
        count_x = 0
        count_o = 0
        count_x_n = 0
        count_o_n = 0
        
        ## check if new state consists of all objects from old state
        ## if everything is ok, count the objects both in old and new state
        for i in range(9):
            if old_state[i] == 'x' or old_state == 'o':
                if not new_state[i] == old_state[i]:
                    return False
            if old_state[i]=='x':
                count_x+=1
            elif old_state[i]=='o':
                count_o+=1
            if new_state[i]=='x':
                count_x_n+=1
            elif new_state[i]=='o':
                count_o_n+=1
        ## if robot is playing with crosses, there should be increase in number of noughts
        if mode == 'x':
            if count_x_n == count_x and count_o_n == count_o + 1:
                return True
        ## if robot is playing with noughts, there should be increase in number of crosses
        elif mode == 'o':
            if count_o_n == count_o and count_x_n == count_x + 1:
                return True
        ## TODO: check if this can be removed
        if count_x_n == 5 and count_o_n == 4:
            return True
        
        ## default return
        return False
      
    def calculateGoalPos(self):
        '''
        Based on the current state of the game, use strategy rules to calculate the desired box on the field
        Then, calculate the 3D position for the tip of the arm in FRAME_ROBOT and which arm to use
        '''
        
        ## calculate the goal box
        if self.mode == 'x':
            [i,j] = strategija_x(self.board)
            self.goal = i*3+j
        else:
            [i,j] = strategija_o(self.board)
            self.goal = i*3+j
        
        ## obtain local transformation matrix from field points
        T_local = np.eye(4, 4, dtype=np.float64)
        T_local[0,3] = self.fieldPoints[0, self.goal]
        T_local[1,3] = self.fieldPoints[1, self.goal]
        T_local[2,3] = self.fieldPoints[2, self.goal]
        
        ## calculate transformation in FRAME_ROBOT
        T_global = np.dot(self.T_field, T_local)
        
        ## obtain 3D point from translation vector
        result = np.zeros((3,1), dtype=np.float64)
        result[0,0]=T_global[0,3]
        result[1,0]=T_global[1,3]
        result[2,0]=T_global[2,3]
        print("[INFO ] Calculated goal position of box %s: [%s, %s, %s]" % (self.goal+1, result[0,0], result[1,0], result[2,0]))
        
        ## calculate which arm to use
        nameEffector = 'RArm'
        nameHand = 'RHand'
        if self.goal == 0 or self.goal == 3 or self.goal == 6:
            nameEffector = 'LArm'
            nameHand = 'LHand'
        print("[INFO ] Using %s" % nameHand)
        
        ## Update the state of the game
        self.state[i*3+j] = self.mode
        if self.mode == 'x':
            self.board[i][j] = 1
        else:
            self.board[i][j] = 0

        return result, nameEffector, nameHand
    
    def playMove(self, nextMove):
        '''
        Moves nameHand with nameEffector to the goalPos using proxy to ALMotion
        TODO: rewrite to utilize PositionInterpolation method from ALMotion
        '''
        
        ## unpack next move
        goalPos = nextMove[0]
        nameEffector = nextMove[1]
        nameHand = nextMove[2]
        print("[INFO ] Put object in %s and touch arm tactile sensor" % nameHand)
        ## open hand
        self.motion.openHand(nameHand)
        ## wait for the object to be placed in the hand
        while True:
            if nameEffector == 'RArm':
                val1 = self.memory.getData("HandRightLeftTouched")
                val2 = self.memory.getData("HandRightBackTouched")
                val3 = self.memory.getData("HandRightRightTouched")
            else:
                val1 = self.memory.getData("HandLeftLeftTouched")
                val2 = self.memory.getData("HandLeftBackTouched")
                val3 = self.memory.getData("HandLeftRightTouched")   
            if val1 or val2 or val3:
                break
        ## close hand
        self.motion.closeHand(nameHand)
        
        ## enable control of the arm
        self.motion.wbEnableEffectorControl(nameEffector, True)
        
        ## extract current position and elevate the hand
        currPos = self.motion.getPosition(nameEffector,2, True)
        currPos[0] = currPos[0]+0.03
        currPos[2] = currPos[0]+0.2
        self.motion.closeHand(nameHand)
        self.motion.setStiffnesses(nameEffector, 1.0)
        
        for i in range(5):
            time.sleep(0.3)
            self.motion.wbSetEffectorControl(nameEffector, currPos[:3])
        
        ## extract goal position and move arm towards it
        goalPosition = [goalPos[0,0], goalPos[1,0], goalPos[2,0]+self.height, 0.0, 0.0, 0.0]
        
        for i in range(5):
            self.motion.wbSetEffectorControl(nameEffector, goalPosition[:3])
            time.sleep(0.3)
        
        ## open hand to release the object
        time.sleep(0.5)
        self.motion.openHand(nameHand)
        time.sleep(0.5)
        
        ## obtain current postion and elevate the arm        
        currPos = self.motion.getPosition(nameEffector,2, True)
        currPos[2] = currPos[2]+0.1
        for i in range(5):
            self.motion.wbSetEffectorControl(nameEffector, currPos[:3])
            time.sleep(0.3)
        time.sleep(0.3)
        
        ## return to safe position
        if nameEffector == 'RArm':
            for i in range(5):
                self.motion.wbSetEffectorControl(nameEffector, right_safe[:3])
                time.sleep(0.3)
        else:
            for i in range(5):
                self.motion.wbSetEffectorControl(nameEffector, left_safe[:3])
                time.sleep(0.3)
        time.sleep(0.5)
        
        ## disable whole body control
        self.motion.wbEnableEffectorControl(nameEffector, False)
        ## put hands in specific position, useful for easier object placement
        ## TODO: remove hard coding
        if nameEffector == 'RArm':
            angles = self.motion.getAngles("RArm", True)
            angles[4]=1.5385600328445435
            self.motion.setAngles('RArm', angles, 0.5)
        else:
            angles = self.motion.getAngles("LArm", True)
            angles[4]=-1.4496722221374512
            self.motion.setAngles('LArm', angles, 0.5)
        time.sleep(1)
        ## close hand and release stiffnesses
        self.motion.closeHand(nameHand)
        self.motion.setStiffnesses("RArm", 0.0)
        self.motion.setStiffnesses("LArm", 0.0)
        
        ## move was executed, exit 
        return

    def play(self):
        '''
        Method for NAO to play the game of noughts and crosses
        Returns False if game is over
        '''
        
        ## find the field
        while not self.findField():
            ## if field was not found, warn
            print("[WARN ] Field not found")
            self.drawstuff(False)
        
        ## if the field was found, display distance
        print("[INFO ] Distance: [%s, %s, %s]" % (self.T_field[0,3], self.T_field[1,3], self.T_field[2,3]))
        ## when the field is found, update the state of the game
        self.state, self.board = self.imgproc.getGameState(self.img, self.lines)
        print("[INFO ] Game state %s" % (self.state))
        
        ## drawstuff
        self.drawstuff(True)
        
        ## check if the game is over when the opponent played
        res = self.checkResult()
        print("[INFO ] Result %s" % res)
        
        ## if the game is over, check which outcome happened and exit
        if res==2:
            self.tts.say("Looks like you beat me. Congratulations")
            return False
        elif res==0:
            self.tts.say("Looks like a draw. Congratulations")
            return False
        
        ## if the game is not over, calculate next move
        nextMove = self.calculateGoalPos()
        
        ## play the move then return to the initial position
        self.playMove(nextMove)
        
        ## check if robot wins or the game is draw
        if self.checkResult() < 0:
            ## if the game is not over, wait for the opponent to play
            while True:
                time.sleep(2)
                while not self.findField():
                    print("[WARN ] Field not found")
                    self.drawstuff(False)
                new_state, new_board = self.imgproc.getGameState(self.img, self.lines)
                
                print("[INFO ] New state %s" % new_state)
                print("[INFO ] Old state %s" % self.state)
                ## check if new state is valid with respect to the old state
                if self.checkValidity(new_state, self.state, self.mode):
                    ## if it is valid, robot needs to play its move
                    break
                else:
                    ## if the state is not valid, robot waits
                    self.tts.say("I am waiting for you to play")
            ## if the loop was broken, the game continues
            return True
        else:
            ## game is over, check the result, say appropriate phrase and return false to indicate that the game is over
            if self.checkResult() == 0:
                self.tts.say("Looks like a draw")
                return False
            else:
                self.tts.say("I win, I win, I win, ha ha ha ha ha ha")
                return False
          
        ## return
        return True
    
    def gameInit(self):
        '''
        Initializes the game, waits for touch on the head to start
        Upon start, detects the state of the game and whose turn it is, assuming that crosses play first
        Returns false if there are too many objects on the playing field
        '''
        ## wait for the front tactile sensor to be touched
        while True:
            fieldFound = self.findField()
            self.drawstuff(fieldFound)
            ## update the state of the game
            self.state, self.board = self.imgproc.getGameState(self.img, self.lines)
            print("[INFO ] Game state %s" % (self.state))
            if fieldFound and self.memory.getData("FrontTactilTouched"):
                break
                
        ## count objects
        count_x = 0
        count_o = 0
        for i in range(9):
            if self.state[i]=='x':
                count_x+=1
            elif self.state[i]=='o':
                count_o+=1
        
        ## if there are 2 or more objects on the playing field, robot does not know how to start
        if count_x + count_o >= 2:
            self.tts.say("I am confused, there are to many objects on the playing field")
            return False
        ## if there is one cross and no noughts, robot assumes the opponent played
        elif count_x == 1 and count_o == 0:
            self.tts.say("Oh, I see you started first. Ok, I will be playing with noughts")
            self.mode = 'o'
            return True
        ## if there are no objects on the playing field, robot assumes that it plays first
        elif count_x == 0 and count_o == 0:
            self.tts.say("Ok, I will play first using crosses")
            self.mode = 'x'
            return True
        
        ## defualt return
        return False
        
        
        
        
        
        
