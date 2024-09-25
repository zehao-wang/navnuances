import numpy as np
from dtw import dtw
from typing import Any, List, Union
from numpy import ndarray
import math

def euclidean_distance(
    pos_a: Union[List[float], ndarray], pos_b: Union[List[float], ndarray]
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)

def ndtw(pred_locations, gt_locations, success_distance=1):
    dtw_distance = dtw(
        pred_locations, gt_locations, dist=euclidean_distance
    )[0]

    nDTW = np.exp(
        -dtw_distance
        / (len(gt_locations) * success_distance)
    )
    return nDTW

def dot_product(v1, v2):
    return np.dot(v1, v2)

def norm(v):
    return np.linalg.norm(v)

def is_projection_inside_segment(A, B, P):
    # Convert points to numpy arrays for vector operations
    A = np.array(A)
    B = np.array(B)
    P = np.array(P)
    
    # Calculate the vector AP
    AP = P - A
    
    # Calculate the vector AB
    AB = B - A
    
    # Calculate the projected point P' on line AB
    if norm(AB) == 0:
        return False

    proj_scale = dot_product(AP, AB) / norm(AB)**2
    proj_point = A + proj_scale * AB
    
    # Check if the projected point lies within the segment AB
    # It lies within the segment if it's not further from A than B is, and vice versa
    return abs(norm(proj_point - A)  + norm(proj_point - B) - norm(AB)) < 1e-7

def check_angle_between_vectors(v1, v2):
    # Convert vectors to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Calculate the dot product of the vectors
    dot_product = np.dot(v1, v2)

    # Compute the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = np.clip(cos_angle, -1,1)

    # Calculate the angle in radians and then convert to degrees
    angle = np.arccos(cos_angle)
    return abs(math.degrees(angle))
