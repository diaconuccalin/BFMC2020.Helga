import socket
import struct
import time
import numpy as np
import datetime
import cv2
import math

from multiprocessing    import Process
from threading          import Thread

from templates.workerprocess import WorkerProcess

class SignDetection(WorkerProcess):
    # ===================================== Worker process =========================================
    def __init__(self, inPs, outPs):
        """Process used for the image processing needed for lane keeping and for computing the steering value.
        
        Parameters
        ----------
        inPs : list(Pipe) 
            List of input pipes (0 - receive image feed from the camera)
        outPs : list(Pipe) 
            List of output pipes (0 - send steering data to the movvement control process)
        """
        super(SignDetection,self).__init__(inPs, outPs)
        
        self.stopCount = 0
        self.parkCount = 0
        self.crossCount = 0
        self.firstCross = True
        
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        super(SignDetection,self).run()

    def _init_threads(self):
        """Initialize the thread.
        """
        if self._blocker.is_set():
            return

        thr = Thread(name='StreamSending',target = self._the_thread, args= (self.inPs[0], self.outPs, ))
        thr.daemon = True
        self.threads.append(thr)


    # ===================================== Custom methods =========================================
    def detectSigns(self, img, outPs):
        """Applies required image processing. 
        
        Parameters
        ----------
        img : Pipe
            The image on which to apply the algorithm.
        """

        # Apply morphologies on color mask
        def prepareMask(img):
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.erode(img, kernel, iterations = 1)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            kernel = np.ones((5, 5), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            kernel = np.ones((11, 11), np.uint8)
            img = cv2.dilate(img, kernel, iterations = 3)

            return img

        # Obtain bounding boxes for blobs in mask that are large enough
        def getBoxes(mask, threshold):
            # Prepare output array
            rectangles = []

            # Obtain contours in image
            ignore, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            # Check if contour is big enough to be relevant
            for cnt in contours:
                if 1000 < cv2.contourArea(cnt):
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    rectangles.append(((int)(x - threshold * w), (int)(y - threshold * h), (int)(w + 2 * threshold * w), (int)(h + 2 * threshold * h)))

            return rectangles

        # Obtain image inside rectangle from original
        def getSigns(rectangles, img):
            # Prepare output array
            signs = []

            # Store the images in the array
            for rectangle in rectangles:
                (x, y, w, h) = rectangle
                signSelection = img[y:(y+h), x:(x+w)]
                signs.append(signSelection)

            return signs

        # Check if np array is empty
        def elements(array):
            return array.ndim and array.size

        # Check if image corresponds to parking sign
        def isParking(sign):
            # Check if sign is not empty
            if(sign is None) or (elements(sign) == 0):
                return 1.0

            # Convert to grayscale and obtain edges
            sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
            sign = cv2.Canny(sign, 5, 200)

            # Load sample and resize it
            sample = cv2.imread("samples/parking.png", 0)
            sample = cv2.resize(sample, (sign.shape[0], sign.shape[0]), interpolation = cv2.INTER_AREA)

            # Obtain contours in sample and sign
            ignore, contours1, hierarchy = cv2.findContours(sign, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ignore, contours2, hierarchy = cv2.findContours(sample, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Default invalid return value
            ret = 1.0

            # Compare the 2 largest contours
            if contours1:
                # Find largest contour from sign
                cntAux = contours1[0]
                for cnt in contours1:
                    if cv2.contourArea(cnt) > cv2.contourArea(cntAux):
                        cntAux = cnt

                # Find largest contour from sample
                cntAux2 = contours2[0]
                for cnt in contours2:
                    if cv2.contourArea(cnt) > cv2.contourArea(cntAux2):
                        cntAux2 = cnt

                # Compare largest shapes
                ret = cv2.matchShapes(cntAux, cntAux2, 1, 0.0)

            return ret

        # Check if image corresponds to crosswalk sign
        def isCrosswalk(sign):
            # Check if sign is not empty
            if(sign is None) or (elements(sign) == 0):
                return 1.0

            # Convert to grayscale and obtain edges
            sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
            sign = cv2.Canny(sign, 5, 200)

            # Load sample and resize it
            sample = cv2.imread("samples/crosswalk.png", 0)
            sample = cv2.resize(sample, (sign.shape[0], sign.shape[0]), interpolation = cv2.INTER_AREA)

            # Apply morphologies
            kernel = np.ones((3, 3), np.uint8)

            sign = cv2.morphologyEx(sign, cv2.MORPH_CLOSE, kernel)
            sample = cv2.morphologyEx(sample, cv2.MORPH_CLOSE, kernel)

            # Obtain contours in sample and sign
            ignore, contours1, hierarchy = cv2.findContours(sign, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ignore, contours2, hierarchy = cv2.findContours(sample, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Default invalid return value
            ret = 1.0

            # Compare the 2 largest contours
            if contours1:
                # Find largest contour from sign
                cntAux = contours1[0]
                for cnt in contours1:
                    if cv2.contourArea(cnt) > cv2.contourArea(cntAux):
                        cntAux = cnt

                # Find largest contour from sample
                cntAux2 = contours2[0]
                for cnt in contours2:
                    if cv2.contourArea(cnt) > cv2.contourArea(cntAux2):
                        cntAux2 = cnt

                # Compare largest shapes
                ret = cv2.matchShapes(cntAux, cntAux2, 1, 0.0)

            return ret

        # Check if image corresponds to stop sign
        def isStop(sign):
            # Check if sign is not empty
            if(sign is None) or (elements(sign) == 0):
                return 1.0

            # Convert to grayscale and obtain edges
            sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
            sign = cv2.Canny(sign, 100, 200)

            # Load sample and resize it
            sample = cv2.imread("samples/stop.png", 0)
            sample = cv2.resize(sample, (sign.shape[0], sign.shape[0]), interpolation = cv2.INTER_AREA)

            # Apply morphology
            kernel = np.ones((4, 4), np.uint8)
            sign = cv2.morphologyEx(sign, cv2.MORPH_DILATE, kernel)

            # Obtain contours in sample and sign
            ignore, contours1, hierarchy = cv2.findContours(sign, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ignore, contours2, hierarchy = cv2.findContours(sample, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Default invalid return value
            ret = 1.0

            # Compare the 2 largest contours
            if contours1:
                # Find largest contour from sign
                cntAux = contours1[0]
                for cnt in contours1:
                    if cv2.contourArea(cnt) > cv2.contourArea(cntAux):
                        cntAux = cnt

                # Find largest contour from sample
                cntAux2 = contours2[0]
                for cnt in contours2:
                    if cv2.contourArea(cnt) > cv2.contourArea(cntAux2):
                        cntAux2 = cnt

                # Compare largest shapes
                ret = cv2.matchShapes(cntAux, cntAux2, 1, 0.0)

            return ret
        
        # Check if image corresponds to parking sign
        def isPriority(sign):
            # Check if sign is not empty
            if(sign is None) or (elements(sign) == 0):
                return 1.0

            # Convert to grayscale, equalize histogram and obtain edges
            sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
            sign = cv2.equalizeHist(sign)
            sign = cv2.Canny(sign, 100, 200)

            # Load sample and resize it
            sample = cv2.imread("samples/parking.png", 0)
            sample = cv2.resize(sample, (sign.shape[0], sign.shape[0]), interpolation = cv2.INTER_AREA)

            # Obtain contours in sample and sign
            ignore, contours1, hierarchy = cv2.findContours(sign, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ignore, contours2, hierarchy = cv2.findContours(sample, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Default invalid return value
            ret = 1.0

            # Compare the 2 largest contours
            if contours1:
                # Find largest contour from sign
                cntAux = contours1[0]
                for cnt in contours1:
                    if cv2.contourArea(cnt) > cv2.contourArea(cntAux):
                        cntAux = cnt

                # Find largest contour from sample
                cntAux2 = contours2[0]
                for cnt in contours2:
                    if cv2.contourArea(cnt) > cv2.contourArea(cntAux2):
                        cntAux2 = cnt

                # Compare largest shapes
                ret = cv2.matchShapes(cntAux, cntAux2, 1, 0.0)

            return ret

        # Sign detection
        # Obtain image dimensions, crop them and update dimensions
        height = img.shape[0]
        width = img.shape[1]

        img = img[0:(int)(height/2), (int)(width/2):width]

        height = img.shape[0]
        width = img.shape[1]

        # Remobe noise
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Obtain hue
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        # Remove noise
        h = cv2.GaussianBlur(h, (5, 5), 0)


        # Create masks for red, blue yellow
        # Superior limit
        ret, r2 = cv2.threshold(h, 255 - 142, 255, cv2.THRESH_BINARY)
        ret, b2 = cv2.threshold(h, 255 - 241, 255, cv2.THRESH_BINARY)
        ret, y2 = cv2.threshold(h, 255 - 162, 255, cv2.THRESH_BINARY)

        # Inferior limit
        h = cv2.bitwise_not(h)

        ret, r1 = cv2.threshold(h, 138, 255, cv2.THRESH_BINARY)
        ret, b1 = cv2.threshold(h, 232, 255, cv2.THRESH_BINARY)
        ret, y1 = cv2.threshold(h, 157, 255, cv2.THRESH_BINARY)

        # Intersection
        r = cv2.bitwise_and(r1, r2)
        b = cv2.bitwise_and(b1, b2)
        y = cv2.bitwise_and(y1, y2)

        # Morphologies on masks
        r = prepareMask(r)
        b = prepareMask(b)
        y = prepareMask(y)


        # Obtain bounding boxes for largest blobs in masks
        redRectangles = getBoxes(r, 0.0)
        blueRectangles = getBoxes(b, 0.0)
        yellowRectangles = getBoxes(y, 0.0)

        # Obtain portions of the original image based on th bounding boxes
        redSigns = getSigns(redRectangles, img)
        blueSigns = getSigns(blueRectangles, img)
        yellowSigns = getSigns(yellowRectangles, img)

        # Make second check based on shape
        for blueSign in blueSigns:
            if isinstance(blueSign, (list, np.ndarray)) and (blueSign is not None) and isParking(blueSign) < 0.25:
                self.parkCount += 1
                if self.parkCount > 3:
                    for outP in outPs:
                        outP.send(0)
                print("Parking")
            if isinstance(blueSign, (list, np.ndarray)) and (blueSign is not None) and isCrosswalk(blueSign) < 0.9:
                self.crossCount += 1
                if self.crossCount > 0 and self.firstCross is True :
                    self.firstCross = False
                    for outP in outPs:
                        outP.send(1)
                print("Crosswalk")
                
        for redSign in redSigns:
            if isinstance(redSign, (list, np.ndarray)) and (redSign is not None) and isStop(redSign) > 10.0:
                self.stopCount += 1
                if self.stopCount > 20:
                    for outP in outPs:
                        outP.send(0)
                
        for yellowSign in yellowSigns:
            if isinstance(yellowSign, (list, np.ndarray)) and (yellowSign is not None) and isPriority(yellowSign) < 0.1:
                print("Priority")

        
    def _the_thread(self, inP, outPs):
        """Obtains image, applies the required image processing and computes the steering angle value. 
        
        Parameters
        ----------
        inP  : Pipe
            Input pipe to read the frames from other process.
        outP : Pipe
            Output pipe to send the steering angle value to other process.
        """
        while True:
            try:
                # Obtain image
                stamps, img = inP.recv()

                # Apply image processing
                self.detectSigns(img, outPs)

            except Exception as e:
                print("Lane keeping error:")
                print(e)
