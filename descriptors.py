#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Descriptor function

Author: Nicolas Ung
MatrNr: 11912380
"""
from typing import List, Tuple

import numpy as np
import cv2

def compute_descriptors(img: np.ndarray,
                        keypoints: List[cv2.KeyPoint],
                        patch_size: int) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """ Calculate a descriptor on patches of the image, centred on the locations of the KeyPoints.

    Calculate a descriptor vector for each keypoint in the list. KeyPoints that are too close to the border to include
    the whole patch are filtered out. The descriptors are returned as a k x m matrix with k being the number of filtered
    KeyPoints and m being the length of a descriptor vector (patch_size**2). The descriptor at row i of
    the descriptors array is the descriptor for the KeyPoint filtered_keypoint[i].

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param keypoints: List of keypoints at which to compute the descriptors
    :type keypoints: List[cv2.KeyPoint]

    :param patch_size: Value defining the width and height of the patch around each keypoint to calculate descriptor.
    :type patch_size: int

    :return: (filtered_keypoints, descriptors):
        filtered_keypoints: List of the filtered keypoints.
            Locations too close to the image boundary to cut out the image patch should not be contained.
        descriptors: k x m matrix containing the patch descriptors.
            Each row vector stores the descriptor vector of the respective corner.
            with k being the number of descriptors and m being the length of a descriptor (usually patch_size**2).
            The descriptor at row i belongs to the KeyPoint at filtered_keypoints[i]
    :rtype: (List[cv2.KeyPoint], np.ndarray)
    """
    ######################################################
    # Write your own code here

    # initialization
    filtered_keypoints = []
    descriptors = []
    half_patch = patch_size // 2
    img_height, img_width = img.shape

    for keypoint in keypoints:
        x, y = int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))

        # check if patch within image boundaries
        left = x - half_patch
        right = x + half_patch + (patch_size % 2)
        top = y - half_patch
        bottom = y + half_patch + (patch_size % 2)

        # do the check and add the descriptor if patch is within image boundaries
        if ((0 <= left) and (right <= img_width) and (0 <= top) and (bottom <= img_height)):
            patch = img[top:bottom, left:right]
            descriptors.append(patch.flatten())
            filtered_keypoints.append(keypoint)

    descriptors = np.array(descriptors)

    ######################################################
    return filtered_keypoints, descriptors



def compute_descriptors2(img: np.ndarray,
                         keypoints: List[cv2.KeyPoint],
                         patch_size: int) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    
    ######################################################
    # Write your own code here

    # initialization
    filtered_keypoints = []
    descriptors = []
    half_patch = patch_size // 2
    img_height, img_width = img.shape

    for keypoint in keypoints:
        x, y = int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))

        # check if patch within image boundaries
        left = x - half_patch
        right = x + half_patch + (patch_size % 2)
        top = y - half_patch
        bottom = y + half_patch + (patch_size % 2)

        # do the check and add the descriptor if patch is within image boundaries
        if ((0 <= left) and (right <= img_width) and (0 <= top) and (bottom <= img_height)):
            patch = img[top:bottom, left:right]
            descriptors.append(patch.flatten())
            filtered_keypoints.append(keypoint)

    descriptors = np.array(descriptors)

    # sort the descriptors and keypoints
    sorted_indices = np.argsort([keypoint.response for keypoint in filtered_keypoints])[::-1]
    descriptors = descriptors[sorted_indices]
    filtered_keypoints = [filtered_keypoints[i] for i in sorted_indices]

    ######################################################
    return filtered_keypoints, descriptors

def compute_descriptors3(img: np.ndarray,
                         keypoints: List[cv2.KeyPoint],
                         patch_size: int) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    
    filtered_keypoints = []
    descriptors = []
    half_patch = patch_size // 2
    img_height, img_width = img.shape

    for keypoint in keypoints:
        x, y = keypoint.pt
        angle = keypoint.angle

        # Compute patch coordinates
        left = int(round(x - half_patch))
        right = int(round(x + half_patch + (patch_size % 2)))
        top = int(round(y - half_patch))
        bottom = int(round(y + half_patch + (patch_size % 2)))

        # Check if patch is within image boundaries
        if 0 <= left and right <= img_width and 0 <= top and bottom <= img_height:
            # Extract the patch
            patch = img[top:bottom, left:right]

            # Rotate the patch
            center = (patch_size // 2, patch_size // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rotated_patch = cv2.warpAffine(patch, M, (patch_size, patch_size))

            # Flatten and append descriptor
            descriptors.append(rotated_patch.flatten())
            filtered_keypoints.append(keypoint)

    descriptors = np.array(descriptors)
    return filtered_keypoints, descriptors