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
from helper_functions import *

if __name__ == '__main__':

    save_image = False  # Enables saving of matches image
    use_matplotlib = False  # Enables saving of matches image
    img_path_1 = 'desk/Image-00.jpg'  # Try different images
    img_path_2 = 'exp1/harry_scale_normal.png'
    img_path_3 = 'exp1/harry_scale_zoomed.png'

    # parameters <<< try different settings!
    sigma1 = 0.8
    sigma2 = 1.5
    threshold = 0.01
    k = 0.04
    patch_size = 5

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

    # show that harris corner detection is not scale invariant
    
    # Load image
    img_gray_2 = cv2.imread(str(current_path.joinpath(img_path_2)), cv2.IMREAD_GRAYSCALE)
    img_gray_2_int = img_gray_2.copy()
    if img_gray_2 is None:
        raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_2)))
    
    # Convert image from uint8 with range [0,255] to float32 with range [0,1]
    img_gray_2 = img_gray_2.astype(np.float32) / 255.
    
    # Harris corner detector on original image
    keypoints_2 = harris_corner(img_gray_2,
                                sigma1=sigma1,
                                sigma2=sigma2,
                                k=k,
                                threshold=threshold)
    
    # Scaled image

    # load image
    img_gray_3 = cv2.imread(str(current_path.joinpath(img_path_3)), cv2.IMREAD_GRAYSCALE)
    img_gray_3_int = img_gray_3.copy()
    if img_gray_3 is None:
        raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_3)))
    
    # Convert image from uint8 with range [0,255] to float32 with range [0,1]
    img_gray_3 = img_gray_3.astype(np.float32) / 255.

    # Harris corner detector on scaled image
    keypoints_3 = harris_corner(img_gray_3,
                                sigma1=sigma1,
                                sigma2=sigma2,
                                k=k,
                                threshold=threshold)
    
    # Draw the keypoints on original image
    keypoints_img_2 = cv2.drawKeypoints(img_gray_2_int, keypoints_2, None)
    show_image(keypoints_img_2, "Harris Corners Scaled", save_image=save_image, use_matplotlib=use_matplotlib)

    # Draw the keypoints on scaled image
    keypoints_img_3 = cv2.drawKeypoints(img_gray_3_int, keypoints_3, None)
    show_image(keypoints_img_3, "Harris Corners Scaled", save_image=save_image, use_matplotlib=use_matplotlib)