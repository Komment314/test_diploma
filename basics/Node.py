class Node():
    def __init__(self, id, joints, parent=None, cost=0):
        self.id = id
        self.joints = joints      
        self.parent = parent
        self.cost = cost

    def get_joints(self):
        return self.joints

    def get_cost(self):
        if self.parent is None:
            return 0
        else:
            return self.cost + self.parent.get_cost()
    
    def __eq__(self, other):
        return (self.joints == other.joints).all()