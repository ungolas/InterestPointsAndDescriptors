#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Corner detection with the Harris corner detector

Author: Nicolas Ung
MatrNr: 11912380
"""
import numpy as np
import cv2

from typing import List

from helper_functions import non_max


def harris_corner(img: np.ndarray,
                  sigma1: float,
                  sigma2: float,
                  k: float,
                  threshold: float) -> List[cv2.KeyPoint]:
    """ Detect corners using the Harris corner detector

    In this function, corners in a grayscale image are detected using the Harris corner detector.
    They are returned in a list of OpenCV KeyPoints (https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html).
    Each KeyPoint includes the attributes, pt (position), size, angle, response. The attributes size and angle are not
    relevant for the Harris corner detector and can be set to an arbitrary value. The response is the result of the
    Harris corner formula.

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: corner threshold
    :type threshold: float

    :return: keypoints:
        corners: List of cv2.KeyPoints containing all detected corners after thresholding and non-maxima suppression.
            Each keypoint has the attribute pt[x, y], size, angle, response.
                pt: The x, y position of the detected corner in the OpenCV coordinate convention.
                size: The size of the relevant region around the keypoint. Not relevant for Harris and is set to 1.
                angle: The direction of the gradient in degree. Relative to image coordinate system (clockwise).
                response: Result of the Harris corner formula R = det(M) - k*trace(M)**2
    :rtype: List[cv2.KeyPoint]

    """
    
    ######################################################
    # Write your own code here
    keypoints = []

    # kernel width from exercise description
    kernel_width1 = np.round( 2 * np.ceil(3 * sigma1) + 1).astype(int)
    kernel_width2 = np.round( 2 * np.ceil(3 * sigma2) + 1).astype(int)

    ########## FROM EXERCISE DESCRIPTION ##########
    kernel1 = cv2.getGaussianKernel(kernel_width1, sigma1)
    gauss1 = np.outer(kernel1, kernel1.transpose())

    kernel2 = cv2.getGaussianKernel(kernel_width2, sigma2)
    gauss2 = np.outer(kernel2, kernel2.transpose())
    ########## FROM EXERCISE DESCRIPTION ##########

    # add gauss and compute the image gradients
    Iy, Ix = np.gradient(cv2.filter2D(img, -1, gauss1))

    # get angles
    angles = np.arctan2(Iy, Ix)

    # autocorrelation matrix with weighted entries
    Ixx = cv2.filter2D((Ix ** 2), -1, gauss2)
    Iyy = cv2.filter2D((Iy ** 2), -1, gauss2)
    Ixy = cv2.filter2D((Ix * Iy), -1, gauss2)

    # calculate Harris values using determinant and trace
    R = Ixx * Iyy - Ixy ** 2 - k * (Ixx + Iyy) ** 2

    # normalizing and setting <0 values to 0
    R[R < 1e-6] = 0.0
    if (np.max(R) > 0):
        R /= np.max(R)

    # non-max suppression
    R_max = non_max(R)

    # finding the corners
    corners = np.where(R_max & (R > threshold))

    # list keypoints from corners in cv2.KeyPoints - format, add angle, response
    keypoints = [cv2.KeyPoint(float(x), float(y), size=1, angle=angles[y, x], response=R[y, x]) for (y, x) in zip(*corners)]

    ######################################################
    return keypoints
