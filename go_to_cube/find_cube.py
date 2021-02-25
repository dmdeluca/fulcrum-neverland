import cv2
import numpy as np
import time

"""
from lab 1 dd
"""
def filter_image(img, hsv_lower, hsv_upper):
    
    # without gaussian blur, many yellows fail:
    blur_amount = 9
    blur = (blur_amount, blur_amount)
    img = cv2.GaussianBlur(img, blur, 0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img, hsv_lower, hsv_upper)
    masked_image = cv2.bitwise_and(img, img, mask=mask)

    # first erode the image to get rid of small specks
    erosion = 15
    erosion_kernel = np.ones((erosion, erosion), np.uint8)
    masked_image = cv2.erode(mask, erosion_kernel)
    # now dilate everything that still exists
    dilation = 45
    dilation_kernel = np.ones((dilation, dilation), np.uint8)
    masked_image = cv2.dilate(masked_image, dilation_kernel)
    # shrink inflated objects back down to size.
    erosion = dilation - erosion
    erosion_kernel = np.ones((erosion, erosion), np.uint8)
    masked_image = cv2.erode(masked_image, erosion_kernel)

    return masked_image
    
"""
from lab 1 dd
"""
def detect_blob(mask):
   
    img = cv2.medianBlur(mask, 7)

    # Set up the SimpleBlobdetector with default parameters with specific values.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = False
    params.filterByConvexity = False
    params.minDistBetweenBlobs = 50
    params.minArea = 100
    params.maxThreshold = 255
    params.thresholdStep = 10
    params.filterByCircularity = False

    # builds a blob detector with the given parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # use the detector to detect blobs.
    keypoints = detector.detect(img)

    # img_with_keypoints = cv2.drawKeypoints(
    #     mask,
    #     keypoints,
    #     outImage=np.array([]),
    #     color=(0, 0, 255),
    #     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    # )
    
    return keypoints

def find_cube(img, hsv_lower, hsv_upper):
    """Find the cube in an image.
        Arguments:
        img -- the image
        hsv_lower -- the h, s, and v lower bounds
        hsv_upper -- the h, s, and v upper bounds
        Returns [x, y, radius] of the target blob, and [0,0,0] or None if no blob is found.
    """
    mask = filter_image(img, hsv_lower, hsv_upper)
    keypoints = detect_blob(mask)

    if keypoints == []:
        return None
    
    keypoints.sort(key = lambda x : -x.size)
    
    
    # img_with_keypoints = cv2.drawKeypoints(
    #     mask,
    #     keypoints,
    #     outImage=np.array([]),
    #     color=(0, 0, 255),
    #     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    # )
    # cv2.imshow("Frame", img_with_keypoints)
    # cv2.waitKey(delay=1)

    #keypoints = sorted(keypoints, key=lambda keypoint: keypoint.size, reverse=True)
    return [keypoints[0].pt[0], keypoints[0].pt[1], keypoints[0].size]

