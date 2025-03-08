import unittest
import numpy as np
from tracker.animal.assignment import GridAssignment  

class TestGridAssignment(unittest.TestCase):
    
    def test_initial_centroid_calculation(self):
        LUT = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        num_animals = 4
        grid = GridAssignment(LUT, num_animals)
        
        expected_centroids = np.array([[0.5, 0.5], [2.5, 0.5], [0.5, 2.5], [2.5, 2.5]])
        np.testing.assert_almost_equal(grid.centroids, expected_centroids)

    def test_update_no_duplicates(self):
        LUT = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        num_animals = 4
        grid = GridAssignment(LUT, num_animals)
        
        # New centroids, no duplicates
        new_centroids = np.array([[0.4, 0.4], [2.4, 0.4], [2.4, 2.4], [0.4, 2.4]])
        updated_centroids = grid.update(new_centroids)
        expected_centroids = np.array([[0.4, 0.4], [2.4, 0.4], [0.4, 2.4], [2.4, 2.4]])
        
        # Check that the centroids are updated correctly
        np.testing.assert_almost_equal(updated_centroids, expected_centroids)

    def test_update_with_duplicates(self):
        LUT = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        num_animals = 4
        grid = GridAssignment(LUT, num_animals)
        
        # New centroids with duplicates
        new_centroids = np.array([[2.4, 0.4], [0.4, 0.4], [0.4, 2.4], [2.4, 2.4], [2.3, 2.6]])  # Duplicate for id=3
        updated_centroids = grid.update(new_centroids)
        
        # Check that the duplicate id=3 centroid is the one closest to the previous position
        expected_centroids = np.array([[0.4, 0.4], [2.4, 0.4], [0.4, 2.4], [2.4, 2.4]])

        np.testing.assert_almost_equal(updated_centroids, expected_centroids)

    def test_update_with_missing_blobs(self):
        LUT = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ])
        num_animals = 4
        grid = GridAssignment(LUT, num_animals)
        
        # New centroids, missing one blob (e.g., no blob for id=3)
        new_centroids = np.array([[0.4, 0.4], [2.4, 0.4], [2.4, 2.4]])
        updated_centroids = grid.update(new_centroids)
        
        # Check that the centroid for the missing blob (id=3) remains at the previous position
        expected_centroids = np.array([[0.4, 0.4], [2.4, 0.4], [0.5, 2.5], [2.4, 2.4]])
        np.testing.assert_almost_equal(updated_centroids, expected_centroids)

    def test_invalid_LUT_outside_range(self):
        LUT = np.array([
            [0, 1, 2],
            [1, 2, 3]
        ])
        num_animals = 3
        with self.assertRaises(ValueError):
            GridAssignment(LUT, num_animals)

    def test_invalid_LUT_not_enough_cell(self):
        LUT = np.array([
            [0, 1, 2],
            [1, 2, 3]
        ])
        num_animals = 6
        with self.assertRaises(ValueError):
            GridAssignment(LUT, num_animals)

if __name__ == "__main__":
    unittest.main()
