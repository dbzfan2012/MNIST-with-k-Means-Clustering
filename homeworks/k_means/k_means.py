from typing import List, Tuple

import numpy as np

from utils import problem


@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """
    n, d = data.shape
    centers = np.zeros((num_centers, d))
    for i in range(num_centers):
        mask = classifications == i
        current_cluster_data = data[mask]
        mean = np.mean(current_cluster_data, axis=0)
        centers[i] = mean
    return centers

@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
    #calculate matrix of distances, for every j and i, ||x_i - u_j||_2^2 = ||x_i||_2^2 + ||u_j||_2^2 - 2x_iu_j
    distances = np.sum(data**2, axis=1)[:, None] + np.sum(centers**2, axis=1) - 2 * (data @ centers.T)
    classifications = np.argmin(distances, axis=1)
    return classifications


def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """
    Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """
    distances = np.zeros((data.shape[0], centers.shape[0]))
    for idx, center in enumerate(centers):
        distances[:, idx] = np.sqrt(np.sum((data - center) ** 2, axis=1))
    return np.mean(np.min(distances, axis=1))


@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3
) -> Tuple[np.ndarray, List[float]]:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Tuple of 2 numpy arrays:
            Element at index 0: Array of shape (num_centers, d) containing trained centers.
            Element at index 1: List of floats of length # of iterations
    """          
    centers = data[:num_centers]
        
    errors = []
    not_converged = True
    while not_converged:
        classifications = cluster_data(data, centers)
        previous_centers = centers.copy()
        centers = calculate_centers(data, classifications, num_centers)
        if np.max(np.abs(centers - previous_centers)) <= epsilon:
            not_converged = False
        errors.append(calculate_error(data, centers))
    return centers, errors
    
    
