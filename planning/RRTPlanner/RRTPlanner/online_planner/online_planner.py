import rclpy, sys
import numpy as np
import random
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer
from tf2_ros.transform_listener import TransformListener
from smarc_mission_msgs.action import GotoWaypoint
from smarc_mission_msgs.msg import Topics as MissionTopics
from RRTPlanner.RRTActionClientNode import DiveToWaypointActionClient
from RRTPlanner.Node_tree import Tree_Node, Tree, copy_tree
from rclpy.executors import MultiThreadedExecutor
from typing import List
import time
import matplotlib.pyplot as plt
import tf_transformations
from dubins_planner.dubins import Waypoint, calc_dubins_path, sample_complete_plan
from RRTPlanner.sam_auv_node import StateInformation

class RRTPlanner():
    def __init__(self,
                 node: Node,
                 goal_tolerance: float = 5.0) -> None:
        self._node = node
        self.obstacle_scale = 10.0  # Scale obstacles by this factor
        # TF listener to get obstacle positions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        # self.tf_buffer.wait_for_transform_async('map_gt', 'sam_auv_v1/cylinder_1_gt', rclpy.time.Time(seconds=15))
        # Action client to send waypoints
        # self._ac = ActionClient(self, GotoWaypoint, f'/sam_auv_v1/{MissionTopics.GOTO_WP_ACTION}')
        self._ac = DiveToWaypointActionClient(self._node)
        self.goal_msg = []

        self.obstacles = []  # Store detected obstacles
        # self.start = (0, 0, 0, random.uniform(-np.pi, np.pi))  # Start position
        self.start = (None, None, None, None)
        self.goal = (65, 5, 0, 0)  # Goal position
        self.goal_tolerance = goal_tolerance  # Goal tolerance radius
        self.rewiring_distance = 3.0  # Rewiring distance
        self.timer = self._node.create_timer(0.1, self.online_manager)  # Timer to get obstacles periodically
        self.obstacles_populated = False  # Flag to check if obstacles are populated
        self.goal_sent_flag = False
        self.tree_initialized = False
        self.path = []
        # self.sam_states = StateInformation(self._node)

        # RRT variables 
        self.tree = None
        self.tree = None
        self.backward_tree = None

        # RRT parameters
        self.stepsize = 8.0
        self.turning_radius = 3.0
        self.step_dubins = 3
        self.original_wp_indices = []

    def get_pose(self):
        """ Get initial orientation of the robot """
        trans = TransformStamped()
        try:
            trans = self.tf_buffer.lookup_transform('sam_auv_v1/base_link_gt', f'sam_auv_v1/odom_gt', rclpy.time.Time(seconds=0))
            posx = trans.transform.translation.x
            posy = trans.transform.translation.y
            quat = trans.transform.rotation
            roll, pitch, yaw = tf_transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            return posx, posy, yaw*180/np.pi
        except Exception as e:
            self._node.get_logger().info(f"Couldn't lookup transform {e}")
            return None, None, None

    def get_obstacles_from_tf(self):
        """ Extract obstacles from the TF tree """
        num_obstacles_gt = 11  # Number of obstacles in the world
        self.obstacles.clear()
        for i in range(1,num_obstacles_gt+1):  # Assuming max 10 obstacles
            
                try:
                    self._node.get_logger().info(f"Trying to get transform again")
                    trans = self.tf_buffer.lookup_transform('sam_auv_v1/odom_gt', f'sam_auv_v1/cylinder_{i}_gt', rclpy.time.Time(seconds=0))
                    trans_radius = self.tf_buffer.lookup_transform(f'sam_auv_v1/cylinder_{i}_gt', f'sam_auv_v1/cylinder_{i}_radius_gt', rclpy.time.Time(seconds = 0))

                    x, y, z = trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z
                    radius = self.obstacle_scale*np.sqrt((trans_radius.transform.translation.x)**2 + (trans_radius.transform.translation.y)**2)
 
                    self.obstacles.append((x, y, z, radius))
                    # self._node.get_logger().info(f"Detected position of obstacle {(x,y)}")
                    # self._node.get_logger().info(f"Detected radius of obstacle {radius}")

                except Exception as e:
                    self._node.get_logger().info(f"Couldn't lookup transform {e}")
                    continue  # Ignore missing obstacles
        if len(self.obstacles) == num_obstacles_gt:
            self.obstacles_populated = True
        self._node.get_logger().info(f"Detected {len(self.obstacles)} obstacles")

    def initialize_tree(self, start):
        """ Initialize the tree with the start node """
        self._node.get_logger().info(f"Initializing Tree with {start}")
        root_node = Tree_Node(state = start)
        self.tree = Tree(root_node, visualize=False)
        self.tree_initialized = True
        # self.visualize_tree(np.array([]),np.array([]),[])
    
    def run_rrt(self, start, goal):
        """ Run RRT algorithm to find a collision-free path """
        # path = [self.start]
        #final_wp is a flag to check if the immediate next wp is the goial, to be used in the online algorithm
        final_wp = False
        # forward_root_node = Tree_Node(state = start)
        # self.visualize = False # Set to True to visualize the tree
        # self.tree = Tree(forward_root_node, visualize=self.visualize) 
        rewire_count = 0
        # self.sam_states.check_state() # this can provide the start position for the this iteration
        start_time = time.time()
        for iter in range(100000):  # Max iterations 
            # self.sam_states.check_state() 
            # self._node.get_logger().info(f"Iteration: {iter}")

            rand_point = self.goal_biased_sampling()
            # self._node.get_logger().info(f"Random point: {rand_point}")
            nearest_node = Tree_Node()
            nearest_node = self.tree.find_nearest_neighbor(rand_point)   # returns node with minimum cost
            # nearest neighbors should be a list hmmm
            path_back = self.find_reverse_path(rand_point)
            
            path_back_exist = len(path_back) > 0
            # self._node.get_logger().info(f"Nearest node: {nearest_node.get_state()} and random point: {rand_point}")
            new_point, path_exist = self.dubins_steer(nearest_node.get_state(), rand_point)  # Steer towards the random point
            # self._node.get_logger().info(f"path exists : {path_exist} and return_path exists : {path_back_exist} ")
            steerable = new_point != nearest_node.get_state()             
            # steerable = new_point != None
            if steerable:
                if path_exist and path_back_exist: # If path exists, add the new node to the tree
                    new_node = Tree_Node(parent = nearest_node, state = new_point)
                    self.tree.add_node(new_node)
                    distance_to_goal = np.linalg.norm(np.array(new_point)[0:2] - np.array(self.goal)[0:2])
                    # self._node.get_logger().info(f"Distance to goal: {distance_to_goal}")
                    if  distance_to_goal < self.goal_tolerance:
                        self._node.get_logger().info("Goal reached")
                        self._node.get_logger().info(f"path length: {len(path_back)}")
                        self.goal, path_to_goal = self.dubins_steer(new_point, self.goal)
                        if path_to_goal:
                            final_node = Tree_Node(parent = nearest_node,state = self.goal)
                            self.tree.add_node(final_node)
                            break
                    rewired = self.rewiring(new_node)
                    rewire_count += 1 if rewired else 0
                    # self._node.get_logger().info(f"Was tree rewired? {rewired}")
        path, cost = self.tree.find_path(final_node)
        self.path = path
        end_time = time.time()
        self._node.get_logger().info(f"Time taken to generate path: {end_time - start_time}")
        self._node.get_logger().info(f"Generated path with {len(path)} waypoints and rewired {rewire_count} times")

        # curr_wp = 8
        # #testing forward tree
        # # forward_tree = copy_tree(path[curr_wp])
        # # final_tree_node = forward_tree.find_nearest_neighbor(final_node.get_state())
        # # path_forward ,cost= forward_tree.find_path(final_tree_node)
        # # dubins_input_forward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_forward 
        # #         if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None]
        # # dubins_out_forward, original_indices = sample_complete_plan(dubins_input_forward, self.turning_radius, self.step_dubins, self.obstacles)
        

        # # # self.visualize_tree(np.array(dubins_out_forward),np.array(dubins_out_backward), path = path_forward, tree = forward_tree)
        # # backward_tree = copy_tree(path[0], final_node = path[curr_wp-2])
        # # final_tree_node = backward_tree.find_nearest_neighbor(final_node.get_state())
        # # self._node.get_logger().info(f"nearest neighbor is : {final_tree_node.get_state()}")
        # # self._node.get_logger().info(f"current waypoint is : {path[curr_wp].get_state()}")
        # # path_backward = self.find_reverse_path(path[curr_wp].get_state(), tree = backward_tree)
        # # self._node.get_logger().info(f"path length: {len(path_backward)}")
        # # dubins_input_backward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_backward
        # #         if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None] 
        # # dubins_out_backward, _ = sample_complete_plan(dubins_input_backward, self.turning_radius, self.step_dubins, self.obstacles)
        # # self.visualize_tree(np.array(dubins_out_forward),np.array(dubins_out_backward), path = path_forward)
        # # final_tree_node = forward_tree.find_nearest_neighbor(final_node.get_state())
        if len(path) == 2:
            #path length being 2 means that the goal is the immediate next waypoint
            final_wp = True
        #now we send this path to dubins planner to get additional waypoints.
        dubins_input_forward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path 
                if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None]
        dubins_out_forward, original_indices = sample_complete_plan(dubins_input_forward, self.turning_radius, self.step_dubins, self.obstacles)
        self._node.get_logger().info(f"path length: {len(path_back)}")
        dubins_input_backward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_back
                if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None] 
        dubins_out_backward, _ = sample_complete_plan(dubins_input_backward, self.turning_radius, self.step_dubins, self.obstacles)

        self._node.get_logger().info(f"the original indices : {original_indices}")


        # dubins_out = dubins_out_forward + dubins_out_backward[::-1]
        # self.visualize_tree(np.array(dubins_out_forward),np.array(dubins_out_backward), path)

        # convert back to pose2D for transfer
        self.original_wp_indices = [int(i) for i in original_indices]
        # give waypoints till the first original index to fawllow
        
        dubins_first_waypoint = dubins_out_forward[0:self.original_wp_indices[1]+1]
        self._node.get_logger().info(f"first waypoint : {dubins_first_waypoint}")
        goal_msg = self.send_waypoints(dubins_first_waypoint, path)

        # #we disconnect the tree from the current node as soon as we are done executing the path by changing the root to the next node
        # self.tree.set_root(path[1])
        return goal_msg, final_wp
    
    def dubins_steer(self, start, end, free_range = False):
        """ Move from start towards end by step_size """
        # self._node.get_logger().info(f"start: {start} end : {end}")
        direction = np.array(end) - np.array(start)
        direction = direction[:2]  # Ignore Z and yaw
        norm = np.linalg.norm(direction)
        if norm == 0:
            return start, False
        
        # new_point = end
        if free_range:
            new_point = end
        else:
            new_point = np.concatenate((min(norm,self.stepsize)* direction / norm, end[2:]))  + np.array(start)

        
        # self._node.get_logger().info(f"projection on the steerable space: {new_point}")
        center1 = start[0:2] + np.array([np.cos(start[3]*np.pi/180 + np.pi/2), np.sin(start[3]*np.pi/180 + np.pi/2)]) * self.turning_radius
        center2 = start[0:2] + np.array([np.cos(start[3]*np.pi/180 - np.pi/2), np.sin(start[3]*np.pi/180 - np.pi/2)]) * self.turning_radius
        # is_front = np.dot(direction, np.array([np.cos(start[3]), np.sin(start[3])])) > 0 
        left_invalid = np.linalg.norm(np.array(new_point)[0:2] - np.array(center2)) < self.turning_radius
        right_invalid = np.linalg.norm(np.array(new_point)[0:2] - np.array(center1)) < self.turning_radius
        # self._node.get_logger().info(f"left valid: {left_invalid} right valid: {right_invalid} is front: {is_front}")
        if left_invalid or right_invalid : 
            # self._node.get_logger().info(f"not in steerable region: {new_point}")
            return start, False
        else: 
            #now we check for collisions and the shortest path for a range of headings
            path_exist,best_heading = self.is_path_collision_free(start,new_point)
            new_point = (new_point[0], new_point[1], new_point[2], best_heading)
            # self._node.get_logger().info(f"returning new point: {new_point}")
            return new_point, path_exist
        

    def is_path_collision_free(self, start, end, optimize_heading = True):
        base_heading = end[3]
        step_heading = 9
        if optimize_heading:
            headings =  base_heading + np.linspace(0, 360, step_heading, endpoint = False) - 180
        else:
            headings = [base_heading]
        min_cost = np.inf
        best_heading = base_heading
        path_exist = False

        for heading in headings:
            wp_1 = Waypoint(start[0], start[1], start[3])
            wp_2 = Waypoint(end[0], end[1], heading)
            path_exist_temp = False
            param = calc_dubins_path(wp_1, wp_2, self.turning_radius, self.obstacles)
            path_exist_temp = (param.seg_final != [0, 0, 0]) # if no path exists this is [0 0 0]
            
            if path_exist_temp:
                path_exist = True
                path_cost = sum(param.seg_final)

                if path_cost < min_cost:
                    min_cost = path_cost 
                    best_heading = heading 
            # else: 
                # print(f"Path does not exist for heading: {heading}")
        # if path_exist:
            # self._node.get_logger().info(f"Best heading: {best_heading - base_heading}")

        return path_exist, best_heading
    
    def rewiring(self, new_node):
        """ Rewire the tree to reduce cost """
        rewired = False
        rewiring_distance = self.rewiring_distance
        for node in self.tree.get_nodes() :
            valid_node = node != new_node and node != self.tree.get_root()
            distance = np.linalg.norm(np.array(new_node.get_state())[0:2] - np.array(node.get_state())[0:2])
            if valid_node and distance < rewiring_distance:
                path_exist = self.is_path_collision_free(new_node.get_state(), node.get_state())  
                
                if  path_exist: #and (distance < self.stepsize)
                    #compute cost of new path and old path
                    # self._node.get_logger().info(f"rewiring node check")
                    old_path,old_cost = self.tree.find_path(node)
                    node.assign_parent(new_node)
                    new_path,new_cost = self.tree.find_path(node)
                    #compare costs
                    if new_cost < old_cost:
                        rewired = True
                        continue
                    else:
                        node.assign_parent(old_path[-2])
        return rewired

    def goal_biased_sampling(self):
        """ Biased sampling towards the goal """
        if random.uniform(0, 1) < 0.1:
            return self.goal
        return (random.uniform(0, 100), random.uniform(-50, 50), 0, random.uniform(-180, 180)) #random.uniform(-np.pi/6, np.pi/6))
    
    def informative_sampling(self, start, goal, cmax_factor= np.inf):
        """ Biased sampling towards the goal """
        if cmax_factor < np.inf:
            cmin = np.linalg.norm((np.array(start)[0:2] - np.array(goal)[0:2]))
            #2 dimensions on our case. heading will be decided by optimizer anyway
            cmax = cmin*cmax_factor
            r1 = cmax/2
            r2 = np.sqrt(cmax**2 - cmin**2)/2
            heading_sample_for_ellipse = random.uniform(-np.pi, np.pi)
            a = random.uniform(0, r1)
            b = random.uniform(0, r2)
            x = a*np.cos(heading_sample_for_ellipse)
            y = b*np.sin(heading_sample_for_ellipse)
            heading_sample = random.uniform(-180, 180)

            theta = np.arctan2(goal[1] - start[1], goal[0] - start[0])  # Angle of line connecting start to goal
            rotated_x = x * np.cos(theta) - y * np.sin(theta)
            rotated_y = x * np.sin(theta) + y * np.cos(theta)
            midpoint = (np.array(start)[0:2] + np.array(goal)[0:2])/2
            # plt.figure(figsize=(8, 8))
            # plt.scatter(midpoint[0], midpoint[1], color='r', marker='x', label="Midpoint")
            # plt.scatter(start[0], start[1], color='g', marker='s', s=150, label="Start")  # Green Square
            # plt.scatter(goal[0], goal[1], color='y', marker='*', s=200, label="Goal")  # Yellow Star
            ## now we plot the elipse
            # theta1 = np.linspace(0, 2*np.pi, 100)
            # x = r1 * np.cos(theta1)
            # y = r2 * np.sin(theta1)
            # x_rot = x * np.cos(theta) - y * np.sin(theta)
            # y_rot = x * np.sin(theta) + y * np.cos(theta)
            # plt.plot(midpoint[0] + x_rot, midpoint[1] + y_rot, color='r', label="Informed Sampling Region")
            # plt.xlabel("X Position")
            # plt.ylabel("Y Position")
            # plt.title("Informed Sampling")
            # plt.legend()
            # plt.grid()
            # # plt.axis("equal")  # Ensures equal scaling for X and Y
            # plt.show()

            return (midpoint[0] + rotated_x, midpoint[1] + rotated_y, 0, heading_sample)
        else:
            return (random.uniform(0, 100), random.uniform(-50, 50), 0, random.uniform(-180, 180)) #random.uniform(-np.pi/6, np.pi/6))
    
    def reconnect_to_tree(self, state ):
        """Reconnects the current measured state to the tree and sets it as the new root."""
        
        # Step 1: Find the nearest node in the tree, maybe it should be path not tree akshually
        # nearest_node = self.tree.find_nearest_neighbor(state)
        final_wp = False
        nearest_node_list = sorted(self.path, key=lambda node: np.linalg.norm(np.array(node.get_state())[0:2] - np.array(state)[0:2]))
        curr_wp = -1
        for curr_wp,nearest_node in enumerate(nearest_node_list) : 
        # Step 2: Try a direct Dubins connection to the nearest node
            path_exists,_ = self.is_path_collision_free(state, nearest_node.get_state(), optimize_heading=False )
            if path_exists:
                break
        if path_exists:
            self._node.get_logger().info("path to tree exists")
            #forward tree upto the nearest node
            #testing forward tree
            forward_tree = copy_tree(nearest_node)
            # Direct connection is possible; create a new root node
            new_root = Tree_Node(state=state, parent=None)
            old_root = forward_tree.get_root()
            old_root.assign_parent(new_root)
            # Update the tree structure
            forward_tree.add_node(new_root)
            forward_tree.set_root(new_root)
            final_node = self.path[-1]
            final_tree_node = forward_tree.find_nearest_neighbor(final_node.get_state())
            path_forward ,cost= forward_tree.find_path(final_tree_node)
            if len(path_forward) == 2:
                #path length being 2 means that the goal is the immediate next waypoint
                final_wp = True
            dubins_input_forward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_forward 
                    if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None]
            dubins_out_forward, original_indices = sample_complete_plan(dubins_input_forward, self.turning_radius, self.step_dubins, self.obstacles)

            #backward path
            backward_tree = copy_tree(self.path[0], final_node = self.path[curr_wp-2])
            path_backward = self.find_reverse_path(state, tree = backward_tree)
            self._node.get_logger().info(f"path length: {len(path_backward)}")
            dubins_input_backward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_backward
                if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None] 
            dubins_out_backward, _ = sample_complete_plan(dubins_input_backward, self.turning_radius, self.step_dubins, self.obstacles)
            self.visualize_tree(np.array(dubins_out_forward),np.array(dubins_out_backward), path = path_forward)
            # convert back to pose2D for transfer
            self.original_wp_indices = [int(i) for i in original_indices]
            # give waypoints till the first original index to fawllow
            dubins_first_waypoint = dubins_out_forward[0:self.original_wp_indices[1]]
            goal_msg = self.send_waypoints(dubins_first_waypoint, path_forward)

            
            return goal_msg, final_wp

            # return True  # Successfully reconnected 
        else: 
            self._node.get_logger().info("Path from current position to the nearest node doesn't exist")
            #what do i give back here
            return  [],final_wp


        # # Step 3: try finding a reverse path\
        # backward_tree = copy_tree(self.path[0], final_node = nearest_node)

        # reverse_path = self.find_reverse_path(state,backward_tree)
        # path_back_exist = len(reverse_path) > 0
        # if not path_back_exist:
        #     return False  # No reconnection possible

        # # Step 4: Reconstruct the tree with the new root and reversed path
        # new_root = reverse_path[0]
        # self.tree.set_root(new_root)

        # for node in reverse_path[1:]:
        #     node.assign_parent(new_root)
        #     self.tree.add_node(node)
        #     new_root = node  # Move down the path


        # self.visualize_tree(np.array(dubins_out_forward),np.array(dubins_out_backward), path = path_forward, tree = forward_tree)
        # backward_tree = copy_tree(self.path[0], final_node = self.path[curr_wp-2])
        # final_tree_node = backward_tree.find_nearest_neighbor(final_node.get_state())
        # self._node.get_logger().info(f"nearest neighbor is : {final_tree_node.get_state()}")
        # self._node.get_logger().info(f"current waypoint is : {self.path[curr_wp].get_state()}")
        # path_backward = self.find_reverse_path(self.path[curr_wp].get_state(), tree = backward_tree)
        # self._node.get_logger().info(f"path length: {len(path_backward)}")
        # dubins_input_backward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_backward
        #         if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None] 
        # dubins_out_backward, _ = sample_complete_plan(dubins_input_backward, self.turning_radius, self.step_dubins, self.obstacles)
        # self.visualize_tree(np.array(dubins_out_forward),np.array(dubins_out_backward), path = path_forward)
        # final_tree_node = forward_tree.find_nearest_neighbor(final_node.get_state())

        # return True  # Successfully reconnected

    def find_reverse_path(self, state, tree = None):
        """ Returns the path from the root to the end node."""
        if tree is None:
            tree = self.tree
        path = []
        reverse_path = []
        nearest_node = tree.find_nearest_neighbor(state)
        # self._node.get_logger().info(f"nearest neighbor is : {nearest_node.get_state()}")
        # self._node.get_logger().info(f"current waypoint is : {state}")
        # nearest_node_list = [nearest_node]
        nearest_node_list = tree.find_nearest_neighbors(state)
        for nearest_node in nearest_node_list:
            if nearest_node == tree.get_root():
                #didnt check dubins here?
                # self._node.get_logger().info(f"nearest node is root")
                reverse_path.append(Tree_Node(state=state, parent=nearest_node))
                return reverse_path
            path, cost = tree.find_path(nearest_node) 
            # self._node.get_logger().info(f"Root of tree : {tree.get_root().get_state()}")
            # self._node.get_logger().info(f"Path length till neighbor : {len(path)}")
            nearest_node_state = nearest_node.get_state()
            nearest_node_reverse_state = nearest_node_state[0], nearest_node_state[1], nearest_node_state[2], nearest_node_state[3] + 180
            first_point,first_path_exist = self.dubins_steer(state, nearest_node_reverse_state, free_range = True)  # Steer towards the parent
            
            new_parent = Tree_Node(state=state, parent=None)
            #right now it gives the first found path, for speeding up forward computation, but we can optimize when we actually try to return?
            if first_path_exist:
                first_node = Tree_Node(state = state, parent = None)
                reversed_node = Tree_Node(state = first_point, parent = first_node)
                reverse_path.append(first_node)
                reverse_path.append(reversed_node)
                new_parent = reversed_node
                # self._node.get_logger().info("first segment is steerable")
                path.append(Tree_Node(state=state, parent=nearest_node))


                for i, p in enumerate(reversed(path)):
                    # reversed_node = Tree_Node(state=(p.get_state()[0], p.get_state()[1], p.get_state()[2], p.get_state()[3] + 180), parent=path[i+1] if i+1 < len(path) else Tree_Node(state=state, parent=None))
                    reversed_node = Tree_Node(state=(p.get_state()[0], p.get_state()[1], p.get_state()[2], p.get_state()[3] + 180), parent=new_parent)
                    parent = reversed_node.get_parent() #pointless
                    
                    # if parent is not None:
                    new_point, path_exist = self.dubins_steer(parent.get_state(), reversed_node.get_state())  # Steer towards the parent
                    reversed_node = Tree_Node(state=new_point, parent=parent)
                    
                    if not path_exist:
                        reverse_path.clear()
                        break

                    # self._node.get_logger().info(f"{i}th segment is steerable")
                    #if path exists, we add it to the list
                    reverse_path.append(reversed_node)
                    new_parent = reversed_node

                # reverse_path = [Tree_Node(state=(p.get_state()[0], p.get_state()[1], p.get_state()[2], -p.get_state()[3]), parent=path[i+1] if i+1 < len(path) else None) for i, p in enumerate(reversed(path))]
            
            if len(reverse_path) > 1:
                # self._node.get_logger().info(f"Reverse path length: {len(reverse_path)}")
                return reverse_path
        # self._node.get_logger().info("No reverse path found")
        reverse_path.clear()
        return reverse_path

    def send_waypoints(self, dub_out, path):
        """Send waypoints sequentially to the action client and plot them with obstacles"""
        list_waypoints = []
        waypoint_array = []  # Store waypoints for plotting

        for waypoint in dub_out:
            #waypoint = np.array(waypoint_node.get_state(), dtype=np.float64)

            goal_waypoint = PoseStamped()
            goal_waypoint.header.frame_id = 'sam_auv_v1/odom_gt'
            goal_waypoint.pose.position.x = float(waypoint[0])
            goal_waypoint.pose.position.y = float(waypoint[1])
            goal_waypoint.pose.position.z = 0.0 # we on surface for now. dubins is 2D
            goal_waypoint.pose.orientation.x = 0.0
            goal_waypoint.pose.orientation.y = 0.0
            goal_waypoint.pose.orientation.z = np.sin(np.pi*waypoint[2]/(2*180.0))
            goal_waypoint.pose.orientation.w = np.cos(np.pi*waypoint[2]/(2*180.0))
            
            list_waypoints.append(goal_waypoint)
            # waypoint_array.append((waypoint[0], waypoint[1]))  # Ignore Z for 2D plotting

        #plot the waypoints and obstacles
        # waypoint_array = np.array(waypoint_array)
        # self.visualize_tree(waypoint_array, path)

        return list_waypoints
        
    def visualize_tree(self, waypoint_array_forward, waypoint_array_backward, path, tree = None):
        """ Plot the waypoints and obstacles """
        plt.figure(figsize=(8, 8))
        if tree is None:
            tree = self.tree
        # Plot waypoints
        if len(waypoint_array_forward) > 0:
            # for i in self.original_wp_indices:
            #     plt.scatter(waypoint_array[i][0], waypoint_array[i][1], color='g', marker='x', label="Original Waypoint" if 'Original Waypoint' not in plt.gca().get_legend_handles_labels()[1] else "")
            x, y = waypoint_array_forward[:, 0], waypoint_array_forward[:, 1]
            plt.plot(x, y, marker='o', linestyle='-', color='b', label="Path")
            plt.scatter(x, y, color='r', label="Waypoints")  # Highlight waypoints
        if len(waypoint_array_backward) > 0:
            x, y = waypoint_array_backward[:, 0], waypoint_array_backward[:, 1]
            plt.plot(x, y, marker='o', linestyle='-', color='r', label  = "Reverse Path")
            plt.scatter(x, y, color='r')
            
        # # Plot obstacles with their radii
        # for ox, oy, _, r in self.obstacles:  # Ignoring Z
        #     obstacle_circle = plt.Circle((ox, oy), r, color='gray', alpha=0.5, fill=True)
        #     plt.gca().add_patch(obstacle_circle)
        #     plt.scatter(ox, oy, color='k', marker='x', label="Obstacle" if 'Obstacle' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        for node in tree.get_nodes():
            if node.get_parent() is not None: 
            #and node not in path:
                parent = node.get_parent()
                plt.plot([node.get_state()[0], parent.get_state()[0]], [node.get_state()[1], parent.get_state()[1]], color='g', linestyle='-', linewidth=0.5)
        # Start and Goal positions
        start_x, start_y = self.start[:2]  # Ignore Z
        goal_x, goal_y = self.goal[:2]  # Ignore Z
        # Plot start and goal
        root = tree.get_root()
        root_state = root.get_state()
        plt.scatter(root_state[0],root_state[1], color='hotpink', marker='s', s=150, label="Root")
        plt.scatter(start_x, start_y, color='g', marker='s', s=150, label="Start")  # Green Square
        plt.scatter(goal_x, goal_y, color='y', marker='*', s=200, label="Goal")  # Yellow Star

        # Plot settings
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Waypoint Path with Obstacles")
        plt.legend()
        plt.grid()
        plt.axis("equal")  # Ensures equal scaling for X and Y
        plt.show()

    def online_manager(self):
        """ Periodically check for obstacles and run RRT """
        start = None
        #get initial orientation
        if not self.obstacles_populated: #will temporarily stay here
            self.get_obstacles_from_tf()
        if not self.get_pose()[0]:
            self._node.get_logger().info("Waiting for initial pose")
            return
        else:
            if self.start[3] is None:
                self.start = (self.get_pose()[0], self.get_pose()[1], 0, self.get_pose()[2]) 
                self._node.get_logger().info(f"Initial pose: {self.start}")
                start = self.start
                self.initialize_tree(self.start)
                #maybe RRT here
                self._node.get_logger().info("Running RRT")
                #first wp
                # self._node.get_logger().info("Sending Goal")

                self.goal_msg, final_waypoint_bool = self.run_rrt(start, self.goal)
                self._ac.waypoint_queue.extend(self.goal_msg[1:])
                self._ac.send_goal()
                self.goal_sent_flag = True
        
        #this takes care of ending the loop when we reach the final waypoint 
        final_waypoint_bool = False #should be handled in the first rrt call but I keep this here for now
        
        # if self.path != [] :
        while not final_waypoint_bool :
            #now we run rrt
            if not self.goal_sent_flag : 
                pose_current = self.get_pose()
                start = (pose_current[0], pose_current[1], 0, pose_current[2])
                # self._node.get_logger().info("Running RRT")
                # self.goal_msg, final_waypoint_bool = self.run_rrt(start, self.goal)
                #maybe instead of running RRT all over again, we find a way back to the tree. read papers???
                self.goal_msg, final_waypoint_bool = self.reconnect_to_tree(start)
                #send goal = first waypoint to action client
                self._node.get_logger().info("Sending Goal")
                #here we need to wait for feedback msg from le client
                #once we have waited, new actual state from tf, and update the state so that rrt is run using it next time. 
                self._ac.waypoint_queue.extend(self.goal_msg[1:])
                self._ac.send_goal()
                self.goal_sent_flag = True

            #where, when and how in the carrying out of the following do I need to cancel if collision is detected????????????????????
            #ok you can run it here I checked yippeeeeee, it works in parallel
            #cancel goal if collision detected HMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
            # self._node.get_logger().info("Checking for collision")
            #chck obstacle and cancel logic and dont forget to tick the bool for sent_goal off.
            # # CHECK FOR COLLISION DURING EXECUTION
            # if self.detect_obstacles():  # Implement this method to check for new obstacles
            #     self._node.get_logger().warn("Obstacle detected! Cancelling goal.")
                
            #     # Cancel the action client goal
            #     self._ac.cancel_goal()
                
            #     # Reset flag so RRT runs again with updated obstacle info
            #     self.goal_sent_flag = False
                
            #     # Recalculate obstacles to re-plan
            #     self.get_obstacles_from_tf()

            #now check if  waypoint_queue is in its last step, if yes, we calculate the next steps.
            if not self._ac.assign_next_waypoint() :
                # new_start = self.get_pose()
                # self.start = new_start
                self._node.get_logger().info("all waypoints traversed. setting flag to done")
                self.goal_sent_flag = False

        #cancel timer
        self._node.get_logger().info("Ending Timer")
        self.timer.cancel()
        # else : 
        #     self._node.get_logger().info("Tree not initialized")
        #     return

def main():
    rclpy.init(args=sys.argv)
    plan_node = rclpy.create_node("rrt_planner")
    planner = RRTPlanner(plan_node)  # Create the planner node

    executor = MultiThreadedExecutor()
    executor.add_node(plan_node)  # Spin planner

    try:
        executor.spin()
         # Keep both nodes alive
    except KeyboardInterrupt:
        planner._node.get_logger().info("Shutting down")
    finally:
        planner._node.destroy_node()
        # ac_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()