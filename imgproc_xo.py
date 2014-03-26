'''
Image processing, v1.0
As of 18.3.2014.

@author: FP
'''

## import image processing definitions
from imgproc_definitions import *

## import OpenCV module
import cv2

## import math funcions
from math import pi, sin, cos, atan2, sqrt

## import numpy for matrix operations and OpenCV
import numpy as np

class ImgProcessingXO():
    ''' 
    Image processing class used by NAO to play tic-tac-toe
    '''
    
    def __init__(self, size, lowCanny=lowerCanny, upCanny=upperCanny, rho=Rho, theta=Theta, houghThresh=houghThreshold, rhoDiv = RhoDivisor, thetaDiv = ThetaDivisor):
        '''
        Used to initialize values for image processing (such as Canny edge detector and Hough lines detector parameters)
        '''
        
        ## set size of the image
        self.height = size[0]
        self.width = size[1]
        ## lower threshold for Canny histeresis procedure
        self.lCanny = lowCanny
        ## upper threshold for Canny histeresis procedure
        self.uCanny = upperCanny
        ## distance resolution of the Hough accumulator (in pixels)
        self.rhoRes = 1.0*rho/rhoDiv
        ## angular resolution of the Hough accumulator (in radians)
        self.thetaRes = 1.0*theta/thetaDiv
        ## threshold for accumulator value to be classified as line
        self.houghThreshold = houghThresh
        
    def preprocessLines(self, image):
        '''
        Uses OpenCv methods to preprocess image: 1) convert image to grayscale;
        2) apply Canny edge detector to extract edges; 3) apply Hough transformation to extract lines
        Smoothing is not used to keep edges of lines sharp for pattern matching
        '''
        ## Convert to grayscale
        self.img_grayscale = cv2.cvtColor(image,cv2.cv.CV_BGR2GRAY)
        ## Extract edges
        edges = cv2.Canny(self.img_grayscale, self.lCanny, self.uCanny)
        ## Apply Hough transformation to obtain lines
	try:
		lines = cv2.HoughLines(edges, self.rhoRes, self.thetaRes, self.houghThreshold)[0]
	except:
		print("No lines found")
		return []
        return lines
    
    def getEndPoints(self, lines):
        ''' 
        Obtain end points of every line in the image. End points are stored in pts1 and pts2 lists, then returned to the calling function
        '''
        
        pts1 = []
        pts2 = []
        flags_line = [0]*len(lines)
        ## for every line
        for i in range(0, len(lines)):
            ## get parameters of the line
            rho=lines[i][0]
            theta = lines[i][1]
            ## check if line was already processed, maybe this can be omitted
            if flags_line[i] == 0:
                ## if line is horizontal
                if theta == 0:
                    pts1.append((rho, 0))
                    pts2.append((rho, self.width))
                ## if line is vertical
                elif theta == pi/2:
                    pts1.append((0, rho))
                    pts2.append((self.height, rho))
                else:
                    ## extract the offset for line in form y=kx+b
                    b = rho/sin(theta)
                    ## calculate four possible points: intersections with lines x=0; y=0; x=height; y=width, only two of them are in the image
                    y1 = rho/sin(theta) #x = 0 -> p1 = (0,y1)
                    x1 = rho/cos(theta) #y = 0 -> p2 = (x1,0)
                    y2 = (rho-cos(theta)*self.height)/sin(theta) #x=height -> p3 = (height, y2)
                    x2 = (rho-sin(theta)*self.width)/cos(theta) #y=width -> p3 = (x2, width)
                    
                    if b<0:
                        ## if b<0 line intersects the y=0 axis in the image (since k must be positive or line would not be in the image)
                        pts1.append((x1,0))
                        ## one point on the edge of the image found, search the other point
                        if y1>=0 and y1<=self.width:
                            ## line cannot intersect both x=0 and y=0 with negative b
                            print("This should never happen")
                            ## TODO: should raise an error or be omitted
                            pts2.append((0, y1))
                        elif x2>=0 and x2<=self.height:
                            ## line intersects both y=0 and y=width edges
                            pts2.append((x2,self.width))
                        else:
                            ## line intersects y=0 and x=height edges
                            pts2.append((self.height, y2))
                    if b>=0 and b<self.width:
                        ## if b is in the image, line intersects the edge x=0
                        pts1.append((0,y1)) ## the same as (0,b)
                        if x1>=0 and x1<=self.height:
                            ## line intersects with y=0 edge
                            pts2.append((x1, 0))
                        elif x2>=0 and x2<=self.height:
                            ## line intersects with y=width edge
                            pts2.append((x2,self.width))
                        else:
                            ## line intersects with x=height edge
                            pts2.append((self.height, y2))
                    if b>=self.width:
                        ## if b is positive and out of the image, line intersects y=width edge
                        pts1.append((x2, self.width))
                        if x1>=0 and x1<=self.height:
                            ## line intersects y=0 edge
                            pts2.append((x1, 0))
                        elif y2>=0 and y2<=self.width:
                            ## line intersects x=height edge
                            pts2.append((self.height,y2))
                        else:
                            ## line intersects x=0 (this should never happen, maybe raise an error)
                            pts2.append((0, y1))
            ## denote that line is processed, maybe can be omitted
            flags_line[i]=1
        ## return calculated end points
        return pts1, pts2
    
    def getLength(self, pt1, pt2):
        '''
        Returns distance between two 2D points
        '''
        val = (pt2[0]-pt1[0])*(pt2[0]-pt1[0])
        val = val + (pt2[1]-pt1[1])*(pt2[1]-pt1[1])
        return sqrt(val)
    
    def getMeanPoint(self, points):
        '''
        Returns mean value for x and y component of points
        '''
        x = []
        y = []
        for point in points:
            x.append(point[0])
            y.append(point[1])
        ## return rounded mean point
        return (round(sum(x)/len(x)), round(sum(y)/len(y)))
    
    def mergeEndPoints(self, lines, RelTol):
        '''
        Merges end lines based on their endpoints in the image. If distance between endings of several lines is smaller than RelTol, lines will be merged
        '''
	## check if there are lines
	if not lines:
		return []
        ## calculate end points for lines
        pts1, pts2 = self.getEndPoints(lines)
        lines_merged = []
        flags_line = [0] * len(pts1)
        ## go through every line
        for i in range(min(len(pts1), len(pts2))):
            pts1_near = []
            pts2_near = []
            ## if line i is not processed
            if flags_line[i] == 0:
                pts1_near.append(pts1[i])
                pts2_near.append(pts2[i])
                ## for every other line after line i
                for j in range(i, min(len(pts1), len(pts2))):
                    ## if line j was not already merged with other line
                    if flags_line[j] == 0:
                        ## if both ends of line j are near to endings of line i, add line j to be merged with line i and flag line j as processed
                        if float(self.getLength(pts1[i], pts1[j]) / self.width) < RelTol and float(self.getLength(pts2[i], pts2[j]) / self.width) < RelTol:
                            pts1_near.append(pts1[j])
                            pts2_near.append(pts2[j])
                            flags_line[j] = 1
                        ## check for different combination of end points, since line may be oriented in a different way
                        elif float(self.getLength(pts1[i], pts2[j]) / self.width) < RelTol and float(self.getLength(pts2[i], pts1[j]) / self.width) < RelTol:
                            pts1_near.append(pts2[j])
                            pts2_near.append(pts1[j])
                            flags_line[j] = 1
                ## all lines that are near are collected, find the mean value of end points
                pt1 = self.getMeanPoint(pts1_near)
                pt2 = self.getMeanPoint(pts2_near)
                ## calculate rho and theta for new merged line
                num = float(pt1[0] - pt2[0])
                den = float(pt2[1] - pt1[1])
                theta_m = atan2(num, den)
                if theta_m < 0:
                    ## shift to [0,2pi]
                    theta_m = theta_m + 2 * pi
                if theta_m == 0: # horizontal line, pt1[0] and pt2[0] should be the same and equal to rho
                    rho_m = float(pt1[0] + pt2[0]) / 2
                elif theta_m == pi / 2: # vertical line, p1[1] and pt2[1] sould be the same and equal to rho
                    rho_m = float(pt1[1] + pt2[1]) / 2
                else:
                    ## line is neither horizontal nor vertical, calculate rho
                    rho_m = sin(theta_m) * (pt1[1] * pt2 [0] - pt1[0] * pt2[1]) / (pt2[0] - pt1[0])
                ## add new line to the list of merged lines
                lines_merged.append((rho_m, theta_m))
                ## flag line i as processed
                flags_line[i] = 1
        ## return merged lines
        return lines_merged
    
    def getIntersections(self,lines):
        ''' 
        Calculates and returns the list of all line intersections that are in the image
        '''
        intersections=[]
        flags_intersections=[0]*len(lines)*len(lines)
        ## if there are less than two lines, there are no intersections
        if len(lines)>1:
            ## go through every combination of two lines
            for i in range(len(lines)):
                for j in range(i, len(lines)):
                    ## if that combination was not checked already, possibly can be omitted
                    if flags_intersections[i*len(lines)+j] == 0:
                        ## get parameters of lines
                        rho1 = lines[i][0]
                        rho2 = lines[j][0]
                        theta1 = lines[i][1]
                        theta2 = lines[j][1]
                        if not theta1 == theta2:
                            ## if lines are not parallel, calculate the intersection
                            den = sin(theta1-theta2)+1e-15 ## add this just in case
                            x_inter = (rho2*sin(theta1)-rho1*sin(theta2))/den
                            y_inter = (rho1*cos(theta2)-rho2*cos(theta1))/den
                            if x_inter > 0 and x_inter<self.height and y_inter > 0 and y_inter < self.width:
                                intersections.append((cv2.cv.Round(x_inter), cv2.cv.Round(y_inter)))
                        ## flag combination as processed, maybe can be omitted
                        flags_intersections[i*len(lines)+j]=1
        ## return calculated intersections
        return intersections
    
    def checkIntersection(self, intersection, image, radius, res, tol):
        '''
        Uses pattern matching to validate the intersection by checking whether there are four black rays spreading from the intersection
        '''
        ## create bins by dividig the circle (360 degrees) with the resolution. Each bin will be checked if it contains the black ray
        bin_rays = [0]*(360/res)
        x = 0
        y = 0
        ## for each ray/bin denoted by its relative angle
        for bin_ray in range(360/res):
            ## check only inside the circle with given radius
            for rho in range(1, radius):
                ## calculate point on the ray
                x = cos(res*bin_ray*3.1415926/180.0) * rho;
                y = sin(res*bin_ray*3.1415926/180.0) * rho;
                x = int(x + intersection[0])
                y = y + int(intersection[1])
                ## if the point on the ray is in the image
                if( x>0 and y>0 and x<self.height and y<self.width):
                    ## sum up the inverted intensities on the ray (finding black rays on white paper)
                    bin_rays[bin_ray] = bin_rays[bin_ray] + (255 - image[y, x])   
        ##find mean value across the rays, this will be a threshold to assess whether the ray is black or white
        max_b = max(bin_rays)
        min_b = min(bin_rays)
        avg = min_b+0.5*(max_b-min_b)
        rays_thresh = [0]*(360/res)
        ## for every bin check if it is white or black
        for i in range(360/res):
            if bin_rays[i]>avg:
                ## there are more black pixels than average, flag as black
                rays_thresh[i]=1
            else:
                ## flag as white
                rays_thresh[i]=0
        ## one-dimensional connected components clustering to merge the rays
        labels = [0]*360
        ## counting how many clusters there are, count value will also be label for each cluster
        count = 0;
        ## for all bins
        for i in range(2*360/res):
            ## in worst case we have to pass all rays twice, due to problem being circular
            if rays_thresh[i%(360/res)]== 0 and rays_thresh[(i-1)%(360/res)]==1 and i>=360/res:
                ## if this ray is white and previous was black and one full circle was completed
                ## then all rays have been processed
                break
            elif rays_thresh[i%(360/res)]== 1 and rays_thresh[(i-1)%(360/res)]==0 and i>=360/res:
                ## if this ray is black and previous was white and one full circle was completed
                ## then all rays have been processed
                break
            if rays_thresh[i%(360/res)]==1:
                ## if this ray is black, check if previous was white
                if rays_thresh[(i-1)%(360/res)]==0:
                    ## if previous ray was white, then the new cluster of black rays was detected
                    ## increase the number of black clusters
                    count = count + 1
                ## label the black ray with the number of the cluster, starting with 1
                labels[i%(360/res)] = count
            else:
                ## white rays have label zero
                labels[i%(360/res)] = 0
        
        ## if there are not exactly four clusters, than there are not four rays spreading from the intersection and it is not valid
        if not count == 4:
            return False        
        
        ## find mean value of the angle for each cluster by using mean of circular values formula mean = atan2(sum(sin(angles))/n,sum(cos(angles))/n)
        sum_sin_angles = [0]*count
        sum_cos_angles = [0]*count
        ## mean value of angles of rays inside the cluster
        rays_angles = [0]*count
        ## for every cluster
        for i in range(1, count+1):
            for j in range(360/res):
                ## for every ray check if it is inside the cluster
                if labels[j]==i and labels[j]>0:
                    ## calculate sums
                    sum_sin_angles[i-1] = sum_sin_angles[i-1]+sin(j*res*pi/180)
                    sum_cos_angles[i-1] = sum_cos_angles[i-1]+cos(j*res*pi/180)
            ## calculate mean value of angle
            rays_angles[i-1] = atan2(sum_sin_angles[i-1]/count, sum_cos_angles[i-1]/count)*180/pi
        
        ## if the intersection is valid, there should be two pairs of rays with relative angle of one with respect to other of arround 180 degrees
        ## due to distortion of the camera and calculation with integers, some tolerance must be introduced 
        if (abs((rays_angles[0]+180)%360 - (rays_angles[2])%360) < tol) or (abs((rays_angles[2]+180)%360 - (rays_angles[0])%360) < tol):
            if (abs((rays_angles[1]+180)%360 - (rays_angles[3])%360) < tol) or (abs((rays_angles[3]+180)%360 - (rays_angles[1])%360) < tol):
                ## if there are two pairs of rays with relative angles of 180 degrees, intersection is valid
                return True
        ## return False if the above check fails
        return False
    
    def indexIntersections(self, intersections):
        '''
        Indexes (sorts) intersections in a counter-clockwise manner, with the first intersection being the one closest to the origin of the image
        '''
        if not intersections:
            ## works for any number of intersections but should raise an error if the list is empty
            return
        ## index of the intersection closest to the origin of the image
        min_index = 0
        ## set min distance to max possible value
        min_dist = self.getLength((0,0), (self.height, self.width))
        ## for all intersections
        for i in range(len(intersections)):
            ## calculate distance to the origin
            dist = self.getLength((0, 0),intersections[i])
            if dist <= min_dist:
                ## set as the closest intersection
                min_index=i
                min_dist=dist  
        ## the other intersections are sorted with respect to the angle relative to the first origin
        ## store angles
        theta=[]
        ## set the first intersection to have large negative value for sorting purposes  
        theta.append(-1e9)
        ## store sorted indexes
        ind=[]
        ind.append(min_index)
        ## for all intersections
        for i in range(len(intersections)):
            ## if ith intersection is not the closest to the origin of the image
            if not i==min_index:
                ## find angle with respect to the intersection closest to the origin of the image
                num = intersections[i][0]-intersections[min_index][0]
                den = intersections[i][1]-intersections[min_index][1]
                theta.append(atan2(num, den))
                ind.append(i)
    
        # sort theta and sort indexes
        for i in range(len(theta)-1):
            for j in range(i, len(theta)):
                if theta[i]>theta[j]:
                    p1 = theta[i]
                    theta[i]=theta[j]
                    theta[j]=p1
                    p1 = ind[i]
                    ind[i]=ind[j]
                    ind[j]=p1
        sortedIntersections=[]
        for i in range(len(ind)):
            sortedIntersections.append(intersections[ind[i]])
        ## returned sorted intersections
        return sortedIntersections
    
    def getIndexedIntersections(self, lines):
        '''
        Used to obtain indexed intersections from a set of merged lines
        '''
        
        ## calculate initial intersections
        intersections = self.getIntersections(lines)
        
        ## check every intersection
        valid_intersections = []
        if not intersections:
            ## return empty list
            return valid_intersections
        
        for intersection in intersections:
            if self.checkIntersection(intersection, self.img_grayscale, 15, 1, 45):
                valid_intersections.append(intersection)
        ## index valid intersections      
        return self.indexIntersections(valid_intersections)
    
    def getLine(self, pt1, pt2):
        '''
        Calculates parameters of the line going through points pt1 and pt2
        Order of points is important since it affects the orientation of the line
        Returns a line as a list of four parameters k, b, rho and theta
        '''
        ## calculate k, probably not neccessary
        k = float(pt2[1]-pt1[1])/(pt2[0]-pt1[0]+1e-15)
        ## calculate b, probably not neccessary
        b = pt1[1]-k*pt1[0]
        ## calculate theta
        num = float(pt1[0]-pt2[0])
        den = float(pt2[1]-pt1[1])
        theta = atan2(num,den)
        ## calculate rho based on the value of theta
        if theta == 0: ## horizontal line
            rho = float(pt1[0]+pt2[0])/2
        elif theta == pi: ## horizontal line with different orientation
            rho = -float(pt1[0]+pt2[0])/2        
        elif theta == pi/2: ## vertical line
            rho = float(pt1[1]+pt2[1])/2
        elif theta == -pi/2: ## vertical line with different orientation
            rho = -float(pt1[1]+pt2[1])/2
        else:
            rho = sin(theta)*(pt1[1]*pt2[0]-pt1[0]*pt2[1])/(pt2[0]-pt1[0])
        
        ## maybe rewrite this in terms of obtaining the same parametes for the different line orientation
        ## that would simplify the update of the state of the game
        return k, b, rho, theta
    
    def getBounds(self, P, cornerPoints):
        '''
        Uses projection matrix and local coordinates of corner points to obtain lines bounding the playing field in the image
        '''
        
        ## calculate image coordinates of corners
        self.corners = np.dot(P, cornerPoints) 
        self.corners = self.corners / self.corners[2]
        
        ## bounding lines through corners
        self.boundingLines = []
        for i in range(4):
            pt2 = (cv2.cv.Round(self.corners[0][(i-1)%4]), cv2.cv.Round(self.corners[1][(i-1)%4]))
            pt1 = (cv2.cv.Round(self.corners[0][i]), cv2.cv.Round(self.corners[1][i]))
            self.boundingLines.append(self.getLine(pt1, pt2))
            
        return
    
    def indexLines(self, indexedIntersections):
        '''
        Used to obtain list of indexed lines from the list of indexed intersections
        Line i is going through intersections i-1 and i (orientation is important)
        '''
        
        ## if there are no intersections, return empty list
        if not indexedIntersections:
            return []
        indexedLines = []
        ## for each intersection
        for i in range(len(indexedIntersections)):
            ## second point is the current intersection
            pt2 = indexedIntersections[i]
            ## first point is previous intersection
            pt1 = indexedIntersections[(i-1)%len(indexedIntersections)]
            ## add line through pt1 and pt2 to the list
            indexedLines.append(self.getLine(pt1, pt2))
        
        ## return list of lines
        return indexedLines
    
    def getContours_O(self, imgHSV):
        '''
        Extracts contours of noughts by applying color segmentation on HSV image
        TODO: remove hard coding of the thresholds
        '''
        ## segment the image
        binaryImg = cv2.inRange(imgHSV, np.asarray(cv2.cv.Scalar(10, 75, 95)), np.asarray(cv2.cv.Scalar(40, 255, 255)))
        ## erode and dilate
        binaryImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        ## find contours
        contours_o = cv2.findContours(binaryImg, cv2.cv.CV_RETR_TREE, cv2.cv.CV_CHAIN_APPROX_SIMPLE)[0]
        ## return
        return contours_o
    
    def getContours_X(self, imgHSV):
        '''
        Extracts contours of crosses by applying color segmentation on HSV image
        TODO: remove hard coding of the thresholds
        '''
        ## segment the image, since crosses are red we need to segment twice
        binaryImg1 = cv2.inRange(imgHSV, np.asarray(cv2.cv.Scalar(155, 60, 65)), np.asarray(cv2.cv.Scalar(180, 255, 255)))
        binaryImg2 = cv2.inRange(imgHSV, np.asarray(cv2.cv.Scalar(0, 60, 65)), np.asarray(cv2.cv.Scalar(20, 255, 255)))
        ## add two binary images
        binaryImg = cv2.add(binaryImg1, binaryImg2)
        ## erode and dilate
        binaryImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        ## find contours
        contours_x= cv2.findContours(binaryImg, cv2.cv.CV_RETR_TREE, cv2.cv.CV_CHAIN_APPROX_SIMPLE)[0]
        ## return
        return contours_x
    
    def isInsideField(self, centroid, boundingLines):
        '''
        Returns true if the centroid is inside the polygon defined by the bounding lines
        '''
        d1 = centroid[0]*cos(boundingLines[0][3])+centroid[1]*sin(boundingLines[0][3])-boundingLines[0][2]
        d2 = centroid[0]*cos(boundingLines[1][3])+centroid[1]*sin(boundingLines[1][3])-boundingLines[1][2]
        d3 = centroid[0]*cos(boundingLines[2][3])+centroid[1]*sin(boundingLines[2][3])-boundingLines[2][2]
        d4 = centroid[0]*cos(boundingLines[3][3])+centroid[1]*sin(boundingLines[3][3])-boundingLines[3][2]
        if d1 < 0 and d2 < 0 and d3 < 0 and d4 < 0 :
            return True        
        return False
    
    def getCentroids(self, img):
        '''
        Returns centroids of objects on the playing field, disregards those objects outside of the field
        '''
        
        ## convert image to HSV color space
        imgHSV = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
        
        ## get contours of noughts
        contours_o = self.getContours_O(imgHSV)
        o = []
        ## find centroids of each nought contour
        if contours_o:
            for contour in contours_o:
                if cv2.contourArea(contour) > 250:
                    ## if contour is big enough, find moments
                    moments = cv2.moments(contour)
                    ## use spatial moments to calculate centroid
                    centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
                    ## add centroid to the list of noughts
                    if self.isInsideField(centroid, self.boundingLines):
                        o.append(centroid)
        
        ## get contours of crosses
        contours_x = self.getContours_X(imgHSV)
        x = []
        ## find centroids
        if contours_x:
            for contour in contours_x:
                if cv2.contourArea(contour) > 250:
                    ## if contour is big enough, find moments
                    moments = cv2.moments(contour)
                    ## use spatial moments to calculate centroid
                    centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
                    ## add centroid to the list of crosses
                    if self.isInsideField(centroid, self.boundingLines):
                        x.append(centroid)
        
        ## return lists of nought and cross centroids
        return o, x
    
    def getBoxInField(self, pt, indexedLines):
        '''
        Used to get the box coordinates of a point, given a set of indexed lines
        '''
        ## if there are lines
        if indexedLines:
            ## and exactly four lines
            if len(indexedLines)==4:
                ## calculate distance from point to the line, with negative distance denoting right and positive left side of the line
                d1 = pt[0]*cos(indexedLines[0][3])+pt[1]*sin(indexedLines[0][3])-indexedLines[0][2]
                d2 = pt[0]*cos(indexedLines[1][3])+pt[1]*sin(indexedLines[1][3])-indexedLines[1][2]
                d3 = pt[0]*cos(indexedLines[2][3])+pt[1]*sin(indexedLines[2][3])-indexedLines[2][2]
                d4 = pt[0]*cos(indexedLines[3][3])+pt[1]*sin(indexedLines[3][3])-indexedLines[3][2]
                if d1 < 0:
                    ## right of line one
                    if d2 < 0:
                        ## right of line two
                        return  (0,0)
                    elif d4 > 0:
                        ## left of the line two, left of the line 4
                        return  (0,1)
                    else:
                        ## left of the line two, right of the line 4
                        return  (0,2)
                elif d2 < 0:
                    ## left of line 1, right of line 2
                    if d3 > 0:
                        ## right of line 3
                        return  (1,0)
                    else:
                        ## left of line 3
                        return  (2,0)
                elif d3 > 0:
                    ## left of line 1, right of line 2, left of line 3
                    if d4 > 0:
                        ## right of line 4
                        return  (1,1)
                    else:
                        ## left of line 4
                        return  (1,2)
                elif d4>0:
                    ## left of line 1, right of line 2, right of line 3, left of line 4
                    return  (2,1)
                else:
                    ## left of line 1, right of line 2, left of line 3, right of line 4
                    return  (2,2) 
        
        ## if all else fails, return empty list       
        return []
    
    def getGameState(self, img, indexedLines):
        '''
        Returns the state of the game by calculating the relations between centroids of noughts/crosses and indexed lines
        '''
        
        ## store the state of the game
        state=['-']*9
        board = [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
        if indexedLines:
            ## if there are indexed lines
            if len(indexedLines)==4:
                ## check that there are exactly four lines and obtain noughts and crosses
                noughts, crosses = self.getCentroids(img)
                if noughts:
                    ## if there are nought, check in which box on the field
                    for nought in noughts:
                        (ind_x,ind_y)=self.getBoxInField(nought, indexedLines)
                        ## update the state of the game
                        state[ind_x*3+ind_y]='o'
                        board[ind_x][ind_y] = 0 
                if crosses:
                    ## if there are crossess, check in which box on the field
                    for cross in crosses:
                        (ind_x, ind_y)=self.getBoxInField(cross, indexedLines)
                        ## update the state of the game
                        state[ind_x*3+ind_y]='x'
                        board[ind_x][ind_y] = 1
        
        ## return the list containing the state of the game
        return state, board       
    
