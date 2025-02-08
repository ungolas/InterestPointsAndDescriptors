#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find the projective transformation matrix (homography) between from a source image to a target image.

Author: Nicolas Ung
MatrNr: 11912380
"""

import numpy as np
from helper_functions import *


def find_homography_ransac(source_points: np.ndarray,
                           target_points: np.ndarray,
                           confidence: float,
                           inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return estimated transforamtion matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the RANSAC algorithm with the
    Least-Squares algorithm to minimize the back-projection error and be robust against outliers.
    Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source (object) image [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target (scene) image [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. Euclidean distance of a point from the transformed point to be considered an inlier
    :type inlier_threshold: float

    :return: (homography, inliers, num_iterations)
        homography: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
        inliers: Is True if the point at the index is an inlier. Boolean array with shape (n,)
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: Tuple[np.ndarray, np.ndarray, int]
    """
    ######################################################
    # Write your own code here
    
    # number of points
    N = source_points.shape[0]

    # size of sample
    m = 4

    # number of iterations
    num_iterations = 0
    max_num_iterations = 1000

    # best inliers
    best_inliers = np.zeros(N, dtype=bool)

    # best homography
    best_suggested_homography = np.eye(3)

    # best number of inliers
    best_inliers_count = 0

    # inlier ratio (=I/N)
    eps = 0

    # break condition from lect. 4 slide 99
    while ((num_iterations <= max_num_iterations) and ((1-eps**m)**num_iterations >= (1-confidence))):

        # 4 random points
        rdm_idxs = np.random.choice(N, m, replace=False)
        rdm_sp = source_points[rdm_idxs]
        rdm_tp = target_points[rdm_idxs]

        # calculate homography
        homography = find_homography_leastsquares(rdm_sp, rdm_tp)

        # add row with 1s
        homogeneous_points = np.concatenate([source_points, np.ones((N, 1))], axis=1)

        # calculate projected points
        projected_points = np.dot(homography, homogeneous_points.T).T

        # normalize, [:, :2] to get only x, y, .reshape(-1, 1) for column vector
        projected_points = projected_points[:, :2] / projected_points[:, 2].reshape(-1, 1)

        # calculate distances
        distances = np.linalg.norm(projected_points - target_points, axis=1)
        inliers = distances < inlier_threshold

        # count inliers
        inliers_count = np.sum(inliers)

        # check if new homography is better
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_inliers = inliers
            best_suggested_homography = homography

            # calculate for termination condition
            eps = inliers_count / N

        # increment iteration counter
        num_iterations += 1

    # end of while loop




    # use least squares to calculate homography with best inliers
    best_suggested_homography = find_homography_leastsquares(source_points[best_inliers], target_points[best_inliers])

    # calculate inliers again with new homography
    homogeneous_points = np.concatenate([source_points, np.ones((N, 1))], axis=1)
    projected_points = np.dot(homography, homogeneous_points.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2].reshape(-1, 1)
    distances = np.linalg.norm(projected_points - target_points, axis=1)
    best_inliers = distances < inlier_threshold


    ######################################################
    return best_suggested_homography, best_inliers, num_iterations


def find_homography_leastsquares(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Return projective transformation matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the Least-Squares algorithm to
    minimize the back-projection error with all points provided. Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source image (object image) as [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target image (scene image) as [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :return: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
    :rtype: np.ndarray with shape (3, 3)
    """
    ######################################################
    # Write your own code here
    if source_points.shape[0] >= 4 and source_points.shape == target_points.shape:
        # N = number of points
        N = source_points.shape[0]

        # x,y coordinates of source points
        x = source_points[:, 0]
        y = source_points[:, 1]

        # x,y coordinates of target points (x',y')
        x_p = target_points[:, 0]
        y_p = target_points[:, 1]

        # lt. Skript fuer gl. Ax=b
        A = np.zeros((2 * N, 8))
        b = np.zeros(2 * N)

        # fill every even and odd row according to formula in script
        A[::2] = np.stack([x, y, np.ones(N), np.zeros(N), np.zeros(N), np.zeros(N), -x_p * x, -x_p * y], axis=1)
        A[1::2] = np.stack([np.zeros(N), np.zeros(N), np.zeros(N), x, y, np.ones(N), -y_p * x, -y_p * y], axis=1)

        # fill b with x' and y'
        b[::2] = x_p
        b[1::2] = y_p

        # use least squares to solve
        h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # re-add 1 as h22
        h = np.append(h, 1)

        # reshape to 3x3
        homography = h.reshape(3, 3)
    else:
        homography = np.eye(3)

    ######################################################
    return homography
