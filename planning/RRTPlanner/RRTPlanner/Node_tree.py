import numpy as np
import matplotlib.pyplot as plt
class Tree_Node:
    def __init__(self, parent =  None, cost = 0, state = None):
        self._children = []
        self._parent = parent
        self._state = state
        self._cost = cost
    
    def assign_state(self, state):
        self._state = state  #state is a tuple of coordinates, (x,y,z), and the relevant sensor data. It will decide the next action to take

    def assign_root(self, root):
        self._root = root

    def assign_children(self, children):
        self._children = children

    def assign_parent(self, parent):
        self._parent = parent

    def add_child(self, child):
        self._children.append(child)
    
    def get_parent(self):
        return self._parent
    
    def get_root(self):
        return self._root
    
    def get_children(self):
        return self._children
    
    def get_state(self):
        return self._state

class Tree:
    def __init__(self, root, visualize = False):
        self._root = root
        self._nodes = [root]  # List of all nodes in the tree
        self._edges = []  # Store edges (parent-child connections)
        self._visualize = visualize
        if self._visualize:
            # Initialize Matplotlib plot
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, 100)  # Adjust based on your space
            self.ax.set_ylim(0, 100)
            self.ax.set_title("RRT* Tree Growth")

    def add_node(self, node):
        self._nodes.append(node)
        if node.get_parent():
            self._edges.append((node.get_parent(), node))  # Store edge
        if self._visualize:
            self.visualize_tree()  # Update visualization dynamically

    def get_nodes(self):
        return self._nodes
    
    def get_root(self):
        return self._root

    def cost(self, start, end):
        """ As of not it is just ditsance between two points. Can be modified to include other factors like information gain from parent to child etc;
         in which case the state will have to be expanded to include that information"""
        return np.linalg.norm(np.array(start.get_state()) - np.array(end.get_state()))
    
    def get_cost(self, start, end):
        if start == None:
            return 0
        return self.cost(start, end)
    
    def find_nearest_neighbor(self, state):
        """ Returns the closest node to a given state """
        if not self._nodes:
            return None  # No nodes in the tree
        return min(self._nodes, key=lambda node: np.linalg.norm(np.array(node.get_state())[0:2] - np.array(state)[0:2]))

    def find_nearest_neighbors(self, state):
        """ Returns a sorted list of nodes based on proximity to the given state """
        if not self._nodes:
            return []  # No nodes in the tree
        return sorted(self._nodes, key=lambda node: np.linalg.norm(np.array(node.get_state())[0:2] - np.array(state)[0:2]))

    def visualize_tree(self):
        self.ax.clear()  # Clear previous frame
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-50, 50)
        self.ax.set_title("RRT* Tree Growth")

        # Draw edges (lines between parent-child nodes)
        for parent, child in self._edges:
            parent_pos = np.array(parent.get_state())[:2]  # Ignore z-axis
            child_pos = np.array(child.get_state())[:2]
            self.ax.plot([parent_pos[0], child_pos[0]], 
                         [parent_pos[1], child_pos[1]], 'b-', alpha=0.7)

        # Draw nodes
        for node in self._nodes:
            pos = np.array(node.get_state())[:2]
            self.ax.plot(pos[0], pos[1], 'ro', markersize=3)

        plt.pause(0.01)  # Pause to update plot

    def find_path(self, end, start=None):
        """Returns the path from the root to the end node."""
        if start is None:
            start = self._root
        path = []
        cost_cumulative = 0
        path.append(end)
        parent = end.get_parent()
        cost_cumulative += self.cost(parent, end)

        while parent is not None:
            path.append(parent)
            cost_cumulative += self.get_cost(parent.get_parent(), parent)
            parent = parent.get_parent()
            if parent == end:
                break  # Avoid infinite loops
        path.reverse()

        if path[0] == start:
            return path, cost_cumulative
        else:
            return [], np.inf