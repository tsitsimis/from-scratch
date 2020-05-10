import numpy as np
from scipy.spatial import KDTree


class DBSCAN:
    """
    Implementation of the Density-Based Spatial Clustering of Applications with Noise
    (DBSCAN) algorithm. It uses Euclidean distance and a KD-Tree as the data-structure to
    query neighbour samples.

    Parameters
    ----------
    epsilon : float
        Neighbourhood radius

    min_samples : int
        Minimum number of samples that define a neighbourhood


    Methods
    -------
    fit(X)
        The main part of the DBSCAN algorithm. Scans the input dataset X and for each
        sample its immediately accessible points (neighbourhood) are found. For each
        neighbour the process is repeated and new neighbours are added until there are
        no other accessible points or all neighbours are visited. Points that can't form
        a neighbourhood (non-basic points) are classified as noise/outliers.
    """

    def __init__(self, epsilon, min_samples):
        self.epsilon = epsilon
        self.min_samples = min_samples

        self.m = None
        self.cluster = None
        self.kdtree = None

    def fit(self, X):
        """
        The main part of the DBSCAN algorithm. Scans the input dataset X and for each
        sample its immediately accessible points (neighbourhood) are found. For each
        neighbour the process is repeated and new neighbours are added until there are
        no other accessible points or all neighbours are visited. Points that can't form
        a neighbourhood (non-basic points) are classified as noise/outliers.

        Parameters
        ----------
        X : numpy.ndarray
             Data table with samples to be clustered
        """

        self.m = 0
        self.cluster = np.array([None] * X.shape[0])
        self.kdtree = KDTree(X)

        for i in range(X.shape[0]):
            if self.cluster[i] is not None:
                continue

            neighbour_inds = self.get_accessible_points_inds(i, X)
            if len(neighbour_inds) < self.min_samples:
                self.cluster[i] = 0
                continue

            self.m += 1
            self.cluster[i] = self.m

            while len(neighbour_inds) > 0:
                j = neighbour_inds[0]

                neighbour_inds = neighbour_inds[1:]

                if self.cluster[j] is not None:
                    continue

                self.cluster[j] = self.m

                neighbour_inds_j = self.get_accessible_points_inds(j, X)
                neighbour_inds = neighbour_inds + [e for e in neighbour_inds_j if e not in neighbour_inds]

    def get_accessible_points_inds(self, ind_x, X):
        """
        Queries the KD-Tree to find samples from X within the epsilon-distance neighborhood
        of a given sample x
        """

        x = X[ind_x, :]

        ind_neighbours = self.kdtree.query_ball_point(x, self.epsilon)
        ind_neighbours.remove(ind_x)

        return ind_neighbours
