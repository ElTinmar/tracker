import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from numpy.typing import NDArray

class GridAssignment:
    '''
    Use a Lookup Table and previous position to determine centroid identity. 
    The LUT must contains all integers in [0,num_animals)
    - If exactly one blob is found in one LUT cell, take that position
    - If two blobs or more are found in one LUT cell, 
      keep only the blob closest to previous position.image_fullres
    - If no blob is found in one LUT cell, use previous poisiton.
    This enforces that num_animals centroids are returned whatever
    happens.
    Centroids are initialized to the center of mass of LUT cells.
    '''

    def __init__(self, LUT: NDArray, num_animals: int = 1) -> None:

        self.num_animals = num_animals

        if set(LUT.flatten()) != set(range(num_animals)):
            raise ValueError(f"invalid LUT")
        
        self.LUT = LUT 
        
        # initialize centroids to LUT cell centers 
        self.centroids = np.zeros((num_animals,2))
        for i in range(num_animals):
            y, x = np.where(LUT == i)
            self.centroids[i] = [np.mean(x), np.mean(y)]

    def update(self, centroids: NDArray) -> NDArray:
        """
        returns centroids with shape (num_animals, 2) after removing duplicates,
        and using previous centroid position for missing data.
        """        

        IDs = [int(self.LUT[int(y), int(x)]) for x, y in centroids]
        unique_ids = sorted(set(IDs))

        # look for duplicated IDs and chose the best among those
        idx_to_keep = []
        for id in unique_ids:
            indices = [i for i,x in enumerate(IDs) if x==id]
            if len(indices) > 1: 
                dist = cdist(centroids[indices, :], [self.centroids[id,:]])
                id_shortest_distance = np.argmin(dist)
                idx_to_keep.append(indices[id_shortest_distance])
            else:
                idx_to_keep.append(indices[0])

        new_centroids = self.centroids.copy()
        new_centroids[unique_ids] = centroids[idx_to_keep] 
        self.centroids = new_centroids 

        return self.centroids
    
class LinearSumAssignment:
    '''
    Using the Hungarian algorithm to identify blobs across frames.
    '''

    def __init__(self, distance_threshold, num_animals: int = 1):
        self.ID = None
        self.ID_max = 0
        self.indices = None
        self.centroids = None
        self.distance_threshold = distance_threshold
        self.num_animals = num_animals

    def update(self, centroids: NDArray) -> NDArray:
            # the following events can happen between calls:
            # - same blobs are present 
            # - one or more blobs appeared
            # - one or more blob disappeared
            # - some blob appeared while others disappeared 

            # blob may appear/disappear when:
            # - they get out of the frame
            # - there is a crossing
            # - tracking fails
        
        # pad centroid to right size
        to_pad = self.num_animals - centroids.shape[0]
        if to_pad < 0:
            centroids = centroids[0:self.num_animals,:]
        else:
            centroids = np.pad(centroids,((0,to_pad), (0,0)), constant_values=np.nan)

        if self.centroids is None:
            self.ID = np.arange(self.num_animals)
            self.ID_max = np.max(self.ID)
            self.centroids = centroids
            self.indices = np.arange(self.num_animals)

        else:
            prev = self.centroids[~np.isnan(self.centroids).any(axis=1),:]
            curr = centroids[~np.isnan(centroids).any(axis=1),:]
            dist = cdist(prev, curr)
            r,c = linear_sum_assignment(dist)
            distances = dist[r,c]
            valid = distances < self.distance_threshold

            # valid closest blob can keep their numbers
            new_id = -1*np.ones((self.num_animals,))
            new_id[c[valid]] = self.ID[r[valid]]

            # others are attributed new numbers
            final_id = np.zeros_like(new_id)
            for idx, value in enumerate(new_id):
                
                if np.isnan(centroids[idx,:]).any():
                    final_id[idx] = np.nan #TODO problem with integers and nan
                    continue
                
                if value == -1:
                    self.ID_max = self.ID_max + 1
                    final_id[idx] = self.ID_max
                
                else:
                    final_id[idx] = value

            self.ID = final_id
            self.centroids = centroids
            self.indices = np.arange(self.num_animals)

            
        return self.centroids