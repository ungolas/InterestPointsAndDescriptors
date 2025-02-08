#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 2: Object recognition with SIFT & RANSAC
Matthias Hirschmanner 2024
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at

"""
from pathlib import Path

import numpy as np
import cv2
from find_homography import find_homography_leastsquares, find_homography_ransac
from helper_functions import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # define list of different confidence value pairs with inlier threshold
    confidence_inlier_threshold_pairs = [(0.99, 5.0), (0.85, 5.0), (0.50, 5.0), (0.25, 5.0), (0.99, 10.0), (0.99, 20.0), (0.99, 50.0)]

    # for normal run
    #confidence_inlier_threshold_pairs = [(0.99, 5.0)]

    for ransac_confidence, ransac_inlier_threshold in confidence_inlier_threshold_pairs:

        # calc avg iterations
        avg_iterations = 0

        # number of iterations
        itrs = 50

        # do a few runs with same parameters
        for i in range(itrs):

            # You can change the parameters here. You should not need to change anything else
            image_nr = 1
            save_image = False
            use_matplotlib = False
            debug = False  # <<< change to reduce output when you're done

            #ransac_confidence = 0.85
            #ransac_inlier_threshold = 5.

            # Get path
            current_path = Path(__file__).parent

            # Load images.
            # scene_img  -> image in which we want to detect the object (trainImage in OpenCV)
            # object_img -> image of the object we want to detect (queryImage in OpenCV)
            scene_img = cv2.imread(str(current_path.joinpath("data/image")) + str(image_nr) + ".jpg")
            if scene_img is None:
                raise FileNotFoundError("Couldn't load image in " + str(current_path))
            scene_img_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)  # trainImage

            object_img = cv2.imread(str(current_path.joinpath("data/object.jpg")))
            if object_img is None:
                raise FileNotFoundError("Couldn't load image in " + str(current_path))
            object_img_gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)  # queryImage

            # Get SIFT keypoints_1 and descriptors
            sift = cv2.SIFT_create()
            target_keypoints, target_descriptors = sift.detectAndCompute(scene_img_gray, None)
            source_keypoints, source_descriptors = sift.detectAndCompute(object_img_gray, None)

            # FLANN (Fast Library for Approximate Nearest Neighbors) parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # For each keypoint in object_img get the 3 best matches
            matches = flann.knnMatch(source_descriptors, target_descriptors, k=2)
            matches = filter_matches(matches)

            # Convert keypoints arrays with shape (n, 2) in the OpenCV convention (x,y).
            # source_points[i] is the matching keypoint to target_points[i]
            source_points = np.array([source_keypoints[match.queryIdx].pt for match in matches])
            target_points = np.array([target_keypoints[match.trainIdx].pt for match in matches])

            homography, best_inliers, num_iterations = find_homography_ransac(source_points,
                                                                            target_points,
                                                                            confidence=ransac_confidence,
                                                                            inlier_threshold=ransac_inlier_threshold)
            
            avg_iterations += num_iterations

        print(f"RANSAC with confidence {ransac_confidence} and inlier threshold {ransac_inlier_threshold} needed {avg_iterations/itrs} iterations")

    # for experiment 3.1:
"""    homography = find_homography_leastsquares(source_points, target_points)
    plot_img = draw_rectangles(scene_img, object_img, homography)
    show_image(plot_img, "Final Result without RANSAC", save_image=save_image, use_matplotlib=use_matplotlib)

    transformed_object_img = cv2.warpPerspective(object_img, homography, dsize=scene_img.shape[1::-1])
    scene_img_blend = scene_img.copy()
    scene_img_blend[transformed_object_img != 0] = transformed_object_img[transformed_object_img != 0]
    show_image(scene_img_blend, "Overlay Object without RANSAC", save_image=save_image, use_matplotlib=use_matplotlib)"""
