#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 2: Interest Points and Descriptors
Clara Haider, Matthias Hirschmanner 2024
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
"""

from pathlib import Path

import cv2
import numpy as np

from harris_corner import harris_corner
from descriptors import compute_descriptors
from descriptors import compute_descriptors2
from descriptors import compute_descriptors3
from helper_functions import *

if __name__ == '__main__':

    save_image = False  # Enables saving of matches image
    use_matplotlib = False  # Enables saving of matches image
    img_path_1 = 'desk/Image-00.jpg'  # Try different images

    # parameters <<< try different settings!
    sigma1 = 0.8
    sigma2 = 1.5
    threshold = 0.01
    k = 0.04
    patch_size = 20

    # Load image
    current_path = Path(__file__).parent
    img_gray_1 = cv2.imread(str(current_path.joinpath(img_path_1)), cv2.IMREAD_GRAYSCALE)
    img_gray_1_int = img_gray_1.copy()
    if img_gray_1 is None:
        raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_1)))

    # Convert image from uint8 with range [0,255] to float32 with range [0,1]
    img_gray_1 = img_gray_1.astype(np.float32) / 255.

    # Harris corner detector on original image
    keypoints_1 = harris_corner(img_gray_1,
                                sigma1=sigma1,
                                sigma2=sigma2,
                                k=k,
                                threshold=threshold)

    # Rotate image
    angle = 30
    img_gray_1_rot = rotate_bound(img_gray_1, angle)
    img_gray_1_int_rot = rotate_bound(img_gray_1_int, angle)
    keypoints_1_rot = harris_corner(img_gray_1_rot,
                                    sigma1=sigma1,
                                    sigma2=sigma2,
                                    k=k,
                                    threshold=threshold)

    # Draw the keypoints on original image
    keypoints_img_1 = cv2.drawKeypoints(img_gray_1_int, keypoints_1, None)
    show_image(keypoints_img_1, "Harris Corners", save_image=save_image, use_matplotlib=use_matplotlib)

    # Draw the keypoints on rotated image
    keypoints_img_1_rot = cv2.drawKeypoints(img_gray_1_int_rot, keypoints_1_rot, None)
    show_image(keypoints_img_1_rot, "Harris Corners Rotated", save_image=save_image, use_matplotlib=use_matplotlib)

    # Create descriptors
    filtered_keypoints_1, descriptors_1 = compute_descriptors3(img_gray_1, keypoints_1, patch_size)
    filtered_keypoints_1_rot, descriptors_1_rot = compute_descriptors3(img_gray_1_rot, keypoints_1_rot, patch_size)

    # FLANN (Fast Library for Approximate Nearest Neighbors) parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # For each keypoint in img_gray_1 get the 2 best matches
    matches = flann.knnMatch(descriptors_1.astype(np.float32), descriptors_1_rot.astype(np.float32), k=2)

    filtered_matches = filter_matches(matches)

    matches_img = cv2.drawMatches(img_gray_1_int,
                                  filtered_keypoints_1,
                                  img_gray_1_int_rot,
                                  filtered_keypoints_1_rot,
                                  filtered_matches,
                                  None)

    show_image(matches_img, "Harris Matches", save_image=save_image, use_matplotlib=use_matplotlib)