import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from numpy.typing import NDArray

'''
Assign identity to tracked objects frame after frame
'''

class GridAssignment:
    def __init__(self, LUT):
        self.ID = None
        self.LUT = LUT
        self.centroids = None
        self.idx_to_keep = None

    def update(self, centroids: NDArray):
        IDs = []
        for x, y in centroids:
            IDs.append(int(self.LUT[int(y), int(x)]))
        
        # look for duplicated IDs and chose the best among those
        unique_ids = sorted(set(IDs))
        idx_to_keep = []
        for id in unique_ids:
            indices = [i for i,x in enumerate(IDs) if x==id]
            if len(indices) > 1 and (self.centroids is not None) and (id < self.centroids.shape[0]): 
                dist = cdist(centroids[indices, :], [self.centroids[id,:]])
                id_shortest_distance = np.argmin(dist)
                idx_to_keep.append(indices[id_shortest_distance])
            else:
                idx_to_keep.append(indices[0])

        self.ID = np.array(unique_ids)
        self.centroids = centroids[idx_to_keep,:]
        self.idx_to_keep = idx_to_keep

    def get_ID(self):
        return self.ID
    
    def get_kept_centroids(self):
        return self.idx_to_keep
    
    def get_centroids(self):
        return self.centroids
    
class LinearSumAssignment:
    def __init__(self, distance_threshold):
        self.ID = None
        self.ID_max = 0
        self.idx_to_keep = None
        self.previous_centroids = None
        self.distance_threshold = distance_threshold

    def update(self, centroids):
            # the following events can happen:
            # - same blobs are present 
            # - one or more blobs appeared
            # - one or more blob disappeared
            # - some blob appeared while others disappeared 

            # blob may appear/disappear when:
            # - they get out of the frame
            # - there is a crossing

        if centroids.size == 0:
            return
        
        if self.previous_centroids is None:
            self.ID = np.arange(centroids.shape[0])
            self.ID_max = np.max(self.ID)
        else:
            dist = cdist(self.previous_centroids, centroids)
            r,c = linear_sum_assignment(dist)
            distances = dist[r,c]
            valid = distances < self.distance_threshold
            new_id = -1*np.ones((centroids.shape[0],), dtype=int)
            
            # valid closest blob can keep their numbers
            new_id[c[valid]] = self.ID[r[valid]]
            
            # others are attributed new numbers
            final_id = np.zeros_like(new_id)
            for idx, value in enumerate(new_id):
                if value == -1:
                    self.ID_max = self.ID_max + 1
                    final_id[idx] = self.ID_max
                else:
                    final_id[idx] = value

            self.ID = final_id

        self.previous_centroids = centroids
        self.idx_to_keep = np.arange(centroids.shape[0])
            
    def get_ID(self):
        return self.ID
    
    def get_kept_centroids(self):
        '''for compatibility with grid assignment'''
        return self.idx_to_keep
    
    def get_centroids(self):
        return self.previous_centroids
 