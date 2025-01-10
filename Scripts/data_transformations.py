'''data_transformations.py
Ninh Giang Nguyen
Performs translation, scaling, and rotation transformations on data
CS 251 / 252: Data Analysis and Visualization
Spring 2024

NOTE: All functions should be implemented from scratch using basic NumPy WITHOUT loops and high-level library calls.
'''
import numpy as np


def normalize(data):
    '''Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    '''
    n = data.shape[0]
    m = data.shape[1]
    normalized_data = np.zeros((n, m))

    for i in range(data.shape[1]):
        normalized_data[:, i ] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))
    return normalized_data


def center(data):
    '''Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    '''
    return data - np.mean(data, axis = 0)


def rotation_matrix_3d(degrees, axis='x'):
    '''Make a 3D rotation matrix for rotating the dataset about ONE variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. It does NOT actually PERFORM the rotation!
    '''
    matrix = np.zeros((3, 3))
    if axis == "x":
        matrix[0, 0] = 1
        matrix[1, 1] = np.cos(np.deg2rad(degrees))
        matrix[1, 2] = - np.sin(np.deg2rad(degrees))
        matrix[2, 1] = np.sin(np.deg2rad(degrees))
        matrix[2, 2] = np.cos(np.deg2rad(degrees))
    
    if axis == "z":
        matrix[0, 0] = np.cos(np.deg2rad(degrees))
        matrix[0, 1] = - np.sin(np.deg2rad(degrees))
        matrix[1, 0] = np.sin(np.deg2rad(degrees))
        matrix[1, 1] = np.cos(np.deg2rad(degrees))   
        matrix[2, 2] = 1

    if axis == "y":
        matrix[0, 0] = np.cos(np.deg2rad(degrees))
        matrix[0, 2] = np.sin(np.deg2rad(degrees))
        matrix[1, 1] = 1
        matrix[2, 0] = - np.sin(np.deg2rad(degrees))
        matrix[2, 2] = np.cos(np.deg2rad(degrees))

    return matrix 
