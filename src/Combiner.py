import cv2
import numpy as np
import sys
import os


sys.path.insert(0, os.path.dirname(__file__))

from . import utilities as util
import geometry as gm
import copy

class Combiner:
    def __init__(self,imageList_,dataMatrix_, output="output"):
        '''
        :param imageList_: List of all images in dataset.
        :param dataMatrix_: Matrix with all pose data in dataset.
        :return:
        '''
        self.imageList = []
        self.dataMatrix = dataMatrix_
        self.output_dir = output
        
        os.makedirs(self.output_dir, exist_ok=True)

        detector = cv2.ORB()
        for i in range(0,len(imageList_)):
            
            # originally [::2, ::2, :]
            image = imageList_[i][::6,::6,:] # downsample the image to speed things up. 4000x3000 is huge!
            M = gm.computeUnRotMatrix(self.dataMatrix[i,:])
            correctedImage = gm.warpPerspectiveWithPadding(image,M)
            self.imageList.append(correctedImage) # store only corrected images to use in combination

            # clear source image from memory immediately to save RAM
            imageList_[i] = None
            # periodic garbage collection to free up memory
            if (i + 1) % 5 == 0:
                import gc
                gc.collect()
                print(f"[COMBINER] Processed {i + 1}/{len(imageList_)} images")
            
        self.resultImage = self.imageList[0]
        
        # final cleanup
        del imageList_
        import gc
        gc.collect()
        
    def createMosaic(self):
        for i in range(1,len(self.imageList)):
            print(f"stitching image {i} of {len(self.imageList)-1}")
            self.combine(i)
        return self.resultImage

    def combine(self, index2):
        '''
        :param index2: index of self.imageList and self.kpList to combine with self.referenceImage and self.referenceKeypoints
        :return: combination of reference image and image at index 2
        '''

        #Attempt to combine one pair of images at each step. Assume the order in which the images are given is the best order.
        #This intorduces drift!
        image1 = copy.copy(self.imageList[index2 - 1])
        image2 = copy.copy(self.imageList[index2])

        '''
        Descriptor computation and matching.
        Idea: Align the images by aligning features.
        '''
        detector = cv2.SIFT_create(500) #SURF showed best results
        # detector.extended = True
        gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        ret1, mask1 = cv2.threshold(gray1,1,255,cv2.THRESH_BINARY)
        kp1, descriptors1 = detector.detectAndCompute(gray1,mask1) #kp = keypoints

        gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        ret2, mask2 = cv2.threshold(gray2,1,255,cv2.THRESH_BINARY)
        kp2, descriptors2 = detector.detectAndCompute(gray2,mask2)
        
        # Check if descriptors were found
        if descriptors1 is None or descriptors2 is None:
            print(f"⚠️  Warning: No features detected in image pair {index2-1}-{index2}. Skipping.")
            return self.resultImage

        #Visualize matching procedure.
        keypoints1Im = cv2.drawKeypoints(image1,kp1, None,color=(0,0,255))
        # util.display("KEYPOINTS",keypoints1Im)
        keypoints2Im = cv2.drawKeypoints(image2,kp2, None,color=(0,0,255))
        # util.display("KEYPOINTS",keypoints2Im)

        matcher = cv2.BFMatcher() #use brute force matching
        matches = matcher.knnMatch(descriptors2,descriptors1, k=2) #find pairs of nearest matches
        #prune bad matches
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.55*n.distance:
                    good.append(m)
        matches = copy.copy(good)
        
        # Check if we have enough matches
        if len(matches) < 4:  # Need at least 4 points for affine/homography
            print(f"⚠️  Warning: Only {len(matches)} matches found for image pair {index2-1}-{index2}. Need at least 4. Skipping.")
            return self.resultImage

        #Visualize matches
        # matchDrawing = util.drawMatches(gray2,kp2,gray1,kp1,matches)
        # util.display("matches",matchDrawing)

        #NumPy syntax for extracting location data from match data structure in matrix form
        src_pts = np.float32([ kp2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        # Validate point counts match
        if len(src_pts) != len(dst_pts) or len(src_pts) < 4:
            print(f"⚠️  Warning: Point count mismatch or insufficient points. Skipping image {index2}.")
            return self.resultImage
        
        '''
        Compute Affine Transform
        Idea: Because we corrected for camera orientation, an affine transformation *should* be enough to align the images
        '''
        A, _ = cv2.estimateAffinePartial2D(src_pts,dst_pts) #false because we only want 5 DOF. we removed 3 DOF when we unrotated
        if A is None: #RANSAC sometimes fails in estimateRigidTransform(). If so, try full homography. OpenCV RANSAC implementation for homography is more robust.
            HomogResult = cv2.findHomography(src_pts,dst_pts,method=cv2.RANSAC)
            if HomogResult[0] is None or HomogResult[0] is None:
                print(f"⚠️  Warning: Could not compute transformation for image {index2}. Skipping.")
                return self.resultImage
            H = HomogResult[0]

        '''
        Compute 4 Image Corners Locations
        Idea: Same process as warpPerspectiveWithPadding() excewpt we have to consider the sizes of two images. Might be cleaner as a function.
        '''
        height1,width1 = image1.shape[:2]
        height2,width2 = image2.shape[:2]
        corners1 = np.float32(([0,0],[0,height1],[width1,height1],[width1,0]))
        corners2 = np.float32(([0,0],[0,height2],[width2,height2],[width2,0]))
        warpedCorners2 = np.zeros((4,2))
        for i in range(0,4):
            cornerX = corners2[i,0]
            cornerY = corners2[i,1]
            if A is not None: #check if we're working with affine transform or perspective transform
                warpedCorners2[i,0] = A[0,0]*cornerX + A[0,1]*cornerY + A[0,2]
                warpedCorners2[i,1] = A[1,0]*cornerX + A[1,1]*cornerY + A[1,2]
            else:
                warpedCorners2[i,0] = (H[0,0]*cornerX + H[0,1]*cornerY + H[0,2])/(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
                warpedCorners2[i,1] = (H[1,0]*cornerX + H[1,1]*cornerY + H[1,2])/(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
        allCorners = np.concatenate((corners1, warpedCorners2), axis=0)
        [xMin, yMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
        [xMax, yMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)

        '''Compute Image Alignment and Keypoint Alignment'''
        translation = np.float32(([1,0,-1*xMin],[0,1,-1*yMin],[0,0,1]))
        warpedResImg = cv2.warpPerspective(self.resultImage, translation, (xMax-xMin, yMax-yMin))
        if A is None:
            fullTransformation = np.dot(translation,H) #again, images must be translated to be 100% visible in new canvas
            warpedImage2 = cv2.warpPerspective(image2, fullTransformation, (xMax-xMin, yMax-yMin))
        else:
            warpedImageTemp = cv2.warpPerspective(image2, translation, (xMax-xMin, yMax-yMin))
            warpedImage2 = cv2.warpAffine(warpedImageTemp, A, (xMax-xMin, yMax-yMin))
        self.imageList[index2] = copy.copy(warpedImage2) #crucial: update old images for future feature extractions

        resGray = cv2.cvtColor(self.resultImage,cv2.COLOR_BGR2GRAY)
        warpedResGray = cv2.warpPerspective(resGray, translation, (xMax-xMin, yMax-yMin))

        '''Compute Mask for Image Combination'''
        ret, mask1 = cv2.threshold(warpedResGray,1,255,cv2.THRESH_BINARY_INV)
        mask3 = np.float32(mask1)/255

        #apply mask
        warpedImage2[:,:,0] = warpedImage2[:,:,0]*mask3
        warpedImage2[:,:,1] = warpedImage2[:,:,1]*mask3
        warpedImage2[:,:,2] = warpedImage2[:,:,2]*mask3

        result = warpedResImg + warpedImage2
        #visualize and save result
        self.resultImage = result
        # util.display("result",result)
        
        # save intermediate results
        intermediate_result_path = os.path.join(self.output_dir, f"intermediateResult_{index2}.png")
        cv2.imwrite(intermediate_result_path, result)
        print(f"Intermediate result saved: {intermediate_result_path}")

        return result

