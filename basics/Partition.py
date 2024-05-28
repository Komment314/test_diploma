import numpy as np
from itertools import product

class PartitionCells():
    def __init__(self, n_dims, radius):
        self.n_dims = n_dims
        self.cell_shifts = np.array([list(shift) for shift in product([-1, 0, 1], repeat=n_dims)])
        self.radius = radius
        self.items_container = {}
        
    def put(self, id, coords):
        coords = np.array(coords)
        cell_name = tuple((coords // self.radius).astype(int))
        if cell_name in self.items_container.keys():
            self.items_container[cell_name].append((id, coords))
        else:
            self.items_container[cell_name] = [(id, coords)]
    
    def get_neighbors_id(self, coords):
        neighbors = []
        cell_name = (coords // self.radius).astype(int)
        for new_cell_name_not_tuple in (cell_name + self.cell_shifts):
            new_cell_name = tuple(new_cell_name_not_tuple)
            if new_cell_name in self.items_container.keys():\
                neighbors += self.items_container[new_cell_name]
        neighbors = [item[0] for item in neighbors]
        return neighbors
    
    def change_radius(self, new_radius):
        items_list = []
        for cell_name in self.items_container.keys():
            items_list += self.items_container[cell_name]

        self.radius = new_radius
        self.items_container = {}

        for item in items_list:
            self.put(*item)