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
from dubins_planner.dubins import Waypoint, calc_dubins_path, sample_complete_plan, sample_between_wps
from RRTPlanner.sam_auv_node import StateInformation
from estimator2 import GP2DTrainer
import pandas as pd
from scipy.spatial import cKDTree, KDTree

class RRTPlanner():
    def __init__(self,
                 node: Node,
                 start = None,
                 goal = None,
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
        if start is None:
            self.start = (None, None, None, None)
        else:
            self.start = start
        if goal is None:
            self.goal = (65, 5, 0, 0)  # Goal position
            # self.goal = (15, 0, 0, 0)  # Goal position

        else:
            self.goal = goal
        self.goal_tolerance = goal_tolerance  # Goal tolerance radius
        self.informed_param = np.inf
        
        self.wp_count = 0
        # self.timer = self._node.create_timer(0.1, self.online_manager)  # Timer to get obstacles periodically
        # self.timer_battery = self._node.create_timer(0.1, self.battery_manager)  # Timer to get battery periodically
        # self.timer_eval = self._node.create_timer(0.1, self.eval_manager)  # Timer to get battery periodically
        self.timer_scenario1 = self._node.create_timer(0.1, self.scenario_manager)  # Timer to get battery periodically
        self.obstacles_populated = False  # Flag to check if obstacles are populated
        self.goal_sent_flag = False
        self.tree_initialized = False
        self.path = []
        # should be mapped to tree hmmm
        self.final_node_list = []
        self.final_node = Tree_Node()
        # self.sam_states = StateInformation(self._node)

        # RRT variables 
        self.tree = None
        # self.tree = None
        self.backward_tree = None
        
        # RRT parameters
        self.stepsize = 16.0
        self.turning_radius = 6.0
        self.step_dubins = 1
        self.original_wp_indices = []
        self.trainer = GP2DTrainer(bounds=(0, 100), resolution=20, inducing_count=50, training_iter=700)

        #rewiring params
        self.verbose = False
        self.rewiring_distance = 60.0  # Rewiring distance
        self.rewiring_limit = 10  # Rewiring limit
        self.rewire_count = 0
        self.scale_uncertainty = 50.0

        #hyperparams for learning 
        self.LAMBDA  = 0.1

        # a simple battery state
        self.battery_state = 100.0
        self.battery_start= False

        self.bounds= ((0, 100),(-50, 50))
        self.start_time_wp = None

        self.optimize_path_back = True
        self.SCALE_ARCLENGTH = 0.5

        #train the surrogate model
        # self.trainGP()

    def battery_manager(self):
        """ Update battery status """
        if not self.battery_start:
            self.battery_start = self.goal_sent_flag
        else:
            if self.battery_state > 0.0:
                self.battery_state -= 0.01
            else:
                self._node.get_logger().info("Battery dead")
                self._node.destroy_node()
                rclpy.shutdown()
        
    def get_pose(self):
        """ Get initial orientation of the robot """
        trans = TransformStamped()
        try:
            trans = self.tf_buffer.lookup_transform(f'sam_auv_v1/odom_gt', 'sam_auv_v1/base_link_gt', rclpy.time.Time(seconds=0))
            posx = trans.transform.translation.x
            posy = trans.transform.translation.y
            quat = trans.transform.rotation
            roll, pitch, yaw = tf_transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            return posx, posy, yaw*180.0/np.pi
        except Exception as e:
            self._node.get_logger().info(f"Couldn't lookup transform {e}")
            return None, None, None

    def get_obstacles_from_tf(self):
        """ Extract obstacles from the TF tree """
        num_obstacles_gt = 6 # Number of obstacles in the world
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
            obstacle_positions = np.array([[obs[0], obs[1], obs[2]] for obs in self.obstacles])
            # self.obstacle_tree = cKDTree(obstacle_positions)
            self.obstacle_tree = KDTree(obstacle_positions)


        self._node.get_logger().info(f"Detected {len(self.obstacles)} obstacles")

    def initialize_tree(self, start):
        """ Initialize the tree with the start node """
        self._node.get_logger().info(f"Initializing Tree with {start}")
        root_node = Tree_Node(state = start)
        self.tree = Tree(root_node, visualize=False)
        self.tree_initialized = True
        # self.visualize_tree(np.array([]),np.array([]),[])
    
    def run_rrt(self, start, goal, obstacle_check = False, baseline = False, passage_bias = False, rewire = True):
        """ Run RRT algorithm to find a collision-free path """
        # path = [self.start]
        #final_wp is a flag to check if the immediate next wp is the goial, to be used in the online algorithm
        final_wp = False
        # forward_root_node = Tree_Node(state = start)
        # self.visualize = False # Set to True to visualize the tree
        # self.tree = Tree(forward_root_node, visualize=self.visualize) 
        rewire_count = 0
        self.final_path_back = []
        final_node = None
        # self.sam_states.check_state() # this can provide the start position for the this iteration
        start_time = time.time()
        for iter in range(10000):  # Max iterations 
            # self.sam_states.check_state() 
            # self._node.get_logger().info(f"Iteration: {iter}")

            rand_point = self.goal_biased_sampling(passage_bias = passage_bias)  # Sample a random point in the space)
            # self._node.get_logger().info(f"Random point: {rand_point}")
            nearest_node = Tree_Node()
            nearest_node = self.tree.find_nearest_neighbor(rand_point)   # returns node with minimum cost
            # nearest neighbors should be a list hmmm
            # path_back = self.find_reverse_path(rand_point, obstacle_check = obstacle_check)
            
            # path_back_exist = len(path_back) > 0
            # self._node.get_logger().info(f"Nearest node: {nearest_node.get_state()} and random point: {rand_point}")
            new_point, path_exist, cost, param = self.dubins_steer(nearest_node.get_state(), rand_point, free_range = False, obstacle_check = obstacle_check )  # Steer towards the random point
            path_back = []
            if baseline:
                # For baseline, we don't check for reverse path
                path_back_exist = True
            else:
                path_back = self.find_reverse_path(new_point, optimize=False, obstacle_check = obstacle_check)
                path_back_exist = len(path_back) > 0
            # self._node.get_logger().info(f"cost : {cost}")
            # self._node.get_logger().info(f"path exists : {path_exist} and return_path exists : {path_back_exist} ")
            steerable = new_point != nearest_node.get_state()    
            
            # steerable = new_point != None
            if steerable:
                if path_exist and path_back_exist: # If path exists, add the new node to the tree
                    new_node = Tree_Node(parent = nearest_node,cost = cost, state = new_point, param = param, path_back = path_back)
                    self.tree.add_node(new_node)
                    distance_to_goal = np.linalg.norm(np.array(new_point)[0:2] - np.array(self.goal)[0:2])
                    # self._node.get_logger().info(f"Distance to goal: {distance_to_goal}")
                    if  distance_to_goal < self.goal_tolerance:
                        self._node.get_logger().info("Goal reached")
                        # self._node.get_logger().info(f"path length: {len(path_back)}")
                        # self.goal, path_to_goal, cost = self.dubins_steer(new_point, self.goal)
                        path_to_goal,final_heading, cost, param = self.is_path_collision_free(new_point, self.goal, optimize_heading=True, obstacle_check = obstacle_check)
                        final_state = (self.goal[0], self.goal[1], self.goal[2], final_heading)
                        path_back_final = self.find_reverse_path(final_state, optimize=False, obstacle_check = obstacle_check)
                        path_back_exist = len(path_back_final) > 0
                        if baseline:
                            path_back_exist = True
                        if path_to_goal and path_back_exist:
                            final_node = Tree_Node(parent = nearest_node,cost = cost,state = final_state, param = param, path_back = path_back_final)
                            self.final_node_list.append(final_node)
                            self.tree.add_node(final_node)
                            break
                    # rewired = self.rewiring(new_node)
                    # rewire_count += 1 if rewired else 0
                    # self._node.get_logger().info(f"Was tree rewired? {rewired}")
        if final_node is None:
            self._node.get_logger().info("Goal not reached within max iterations")
            return [], False
        path, cost = self.tree.find_path(final_node)
        # for node in path[1:]:
        #     self._node.get_logger().info(f"Node state: {node.get_param().seg_final}")
        self.path = path
        end_time = time.time()
        self._node.get_logger().info(f"Time taken to generate path: {end_time - start_time}")
        self._node.get_logger().info(f"Generated path with {len(path)} waypoints and rewired {rewire_count} times with cost {cost}")
        
        # #now we send this path to dubins planner to get additional waypoints.
        # dubins_input_forward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path 
        #         if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None]
        # if obstacle_check:
        #     dubins_out_forward, original_indices = sample_complete_plan(dubins_input_forward, self.turning_radius, self.step_dubins, self.obstacles, self.obstacle_tree)
        # else:
        #     dubins_out_forward, original_indices = sample_complete_plan(dubins_input_forward, self.turning_radius, self.step_dubins)
        
        # if not baseline:
        #     path_back = self.find_reverse_path(final_node.get_state(), optimize = True, obstacle_check = obstacle_check)
        #     if len(path_back) == 0:
        #         self._node.get_logger().info(f"PATH BACK NOT FOUND USING OPTIMIZE")
        #         path_back = self.find_reverse_path(final_node.get_state(), optimize = False, obstacle_check = obstacle_check)
        #     self._node.get_logger().info(f"path length: {len(path_back)}")
        #     if len(path_back) == 1:
        #         self._node.get_logger().info(f"path back: {path_back[0].get_state()}")
        #     dubins_input_backward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_back
        #             if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None] 
        #     if obstacle_check:
        #         dubins_out_backward, _ = sample_complete_plan(dubins_input_backward, self.turning_radius, self.step_dubins, self.obstacles, self.obstacle_tree)
        #     else:
        #         dubins_out_backward, _ = sample_complete_plan(dubins_input_backward, self.turning_radius, self.step_dubins)
        # self._node.get_logger().info(f"the original indices : {original_indices}")

        # self.original_wp_indices = [int(i) for i in original_indices]
        # self.visualize_tree(np.array(dubins_out_forward),np.array(dubins_out_backward), path, visualize_back=False)

        if len(path) == 2:
            #path length being 2 means that the goal is the immediate next waypoint
            final_wp = True
        else : 
            if rewire:
                start_time = time.time()
                rewire_iter = 0
                while time.time() - start_time < 15:
                    rewire_iter += 1
                    # self._node.get_logger().info(f"Rewiring iteration: {rewire_iter}")
                    self.informed_rrtstar(eval=False, informed = True ,obstacle_check = obstacle_check, baseline = baseline)
                self._node.get_logger().info(f"Time taken to generate path: {time.time() - start_time}")
                
                self.assign_shortest_path()
                self._node.get_logger().info(f"Generated path with {len(self.path)} waypoints and rewired {self.rewire_count} times")
            # for node in self.tree.get_nodes():
                # self._node.get_logger().info(f"cost : {node.get_cost()}")
        path = self.path
        # for node in path[1:]:
        #     self._node.get_logger().info(f"param: {node.get_param().seg_final} and state: {node.get_state()[3]}")
        # computed_cost, arc_length = self.compute_path_cost(path)
        #now we send this path to dubins planner to get additional waypoints.
        
        dubins_input_forward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path 
                if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None]
        if obstacle_check:
            dubins_out_forward, original_indices = sample_complete_plan(dubins_input_forward, self.turning_radius, self.step_dubins, self.obstacles, self.obstacle_tree)
        else:
            dubins_out_forward, original_indices = sample_complete_plan(dubins_input_forward, self.turning_radius, self.step_dubins)
        self.original_wp_indices = [int(i) for i in original_indices]

        if len(path) == 2:
            #path length being 2 means that the goal is the immediate next waypoint
            final_wp = True
        # if not baseline:
        #     path_back = self.find_reverse_path(self.final_node.get_state(), optimize = True, obstacle_check= obstacle_check)
        #     if len(path_back) == 0:
        #         self._node.get_logger().info(f"PATH BACK NOT FOUND USING OPTIMIZE")
        #         path_back = self.find_reverse_path(final_node.get_state(), optimize = False, obstacle_check = obstacle_check)
        #     self._node.get_logger().info(f"path length: {len(path_back)}")
        #     dubins_input_backward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_back
        #             if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None] 
        #     if obstacle_check:
        #         dubins_out_backward, _ = sample_complete_plan(dubins_input_backward, self.turning_radius, self.step_dubins, self.obstacles, self.obstacle_tree)
        #     else:
        #         dubins_out_backward, _ = sample_complete_plan(dubins_input_backward, self.turning_radius, self.step_dubins)
        #     self.visualize_tree(np.array(dubins_out_forward),np.array(dubins_out_backward), path, visualize_back=True)
        # else:
        #     self.visualize_tree(np.array(dubins_out_forward),np.array([]), path, visualize_back=False)
        # self._node.get_logger().info(f"the original indices : {original_indices}")
        
        # give waypoints till the first original index to fawllow
        # dubins_first_waypoint = dubins_out_forward[0:self.original_wp_indices[1]+1]
        for wp_index,wp in enumerate(dubins_out_forward):
            #here we use the goal tolerance from the action client
            if np.linalg.norm(np.array(wp[0:2]) - np.array(self.start[0:2])) > 5.0:
                # self._node.get_logger().info(f"found start waypoint {wp}")
                break
        
        dubins_first_waypoint = dubins_out_forward[wp_index:self.original_wp_indices[1]+1]

        # self._node.get_logger().info(f"first waypoint : {dubins_first_waypoint[0]}")
        goal_msg = self.send_waypoints(dubins_first_waypoint, path)

        return goal_msg, final_wp

    def informed_rrtstar(self, tree =None, eval = True, informed = True, obstacle_check = False, baseline = False):
        """Informed RRT-star iterations"""
        if tree is None:
            tree = self.tree
        start_node = tree.get_root()
        final_node =  tree.find_nearest_neighbor(self.goal)
        #maybe it shouldn't search this every call
        if informed:
            furthest_node, max_distance = max(
            ((node, 
            np.linalg.norm(np.array(node.get_state())[0:2] - np.array(start_node.get_state())[0:2]) +
            np.linalg.norm(np.array(node.get_state())[0:2] - np.array(final_node.get_state())[0:2]))
            for node in self.path),
            key=lambda x: x[1])
            
            #HYPERPARAMETER FOR UNCERTAINTY
            # LAMBDA = 0.1
            if eval:
                path_uncertainty = self.compute_path_uncertainty(self.path)
                c_best = max_distance + self.LAMBDA * path_uncertainty
            else:
                c_best = max_distance

            #now we rewire the tree
            rand_point = self.informative_sampling(start_node.get_state(), final_node.get_state(), cmax = max_distance)
        else:
            rand_point = self.goal_biased_sampling()

        nearest_node = Tree_Node()
        nearest_node = tree.find_nearest_neighbor(rand_point)   # returns node with minimum cost
        # nearest neighbors should be a list hmmm
        
        # path_back_exist = True
        # self._node.get_logger().info(f"Nearest node: {nearest_node.get_state()} and random point: {rand_point}")
        new_point, path_exist, cost, param = self.dubins_steer(nearest_node.get_state(), rand_point, optimize=self.optimize_path_back, free_range =False, obstacle_check = obstacle_check)  # Steer towards the random point
        path_back = []
        if not baseline:
            path_back = self.find_reverse_path(new_point, tree,optimize=False, obstacle_check = obstacle_check)
            path_back_exist = len(path_back) > 0
        else:
            path_back_exist = True
        # self._node.get_logger().info(f"path exists : {path_exist} and return_path exists : {path_back_exist} ")
        steerable = new_point != nearest_node.get_state()             
        # steerable = new_point != None
        if steerable:
            if path_exist and path_back_exist: # If path exists, add the new node to the tree
                new_node = Tree_Node(parent = nearest_node, cost=cost, state = new_point, param = param, path_back = path_back)
                tree.add_node(new_node)
                distance_to_goal = np.linalg.norm(np.array(new_point)[0:2] - np.array(self.goal)[0:2])
                # self._node.get_logger().info(f"Distance to goal: {distance_to_goal}")
                if  distance_to_goal < self.goal_tolerance:
                    # self._node.get_logger().info("Goal reached")
                    # self._node.get_logger().info(f"path length: {len(path_back)}")
                    path_to_goal,final_heading, cost, param = self.is_path_collision_free(new_point, self.goal, obstacle_check = obstacle_check)
                    final_state = (self.goal[0], self.goal[1], self.goal[2], final_heading)
                    path_back_final = []
                    if not baseline:
                        path_back_final = self.find_reverse_path(final_state, optimize=False, obstacle_check = obstacle_check)
                        path_back_exist = len(path_back_final) > 0
                    else:
                            path_back_exist = True
                    if path_to_goal and path_back_exist:
                        final_node = Tree_Node(parent = nearest_node,cost = cost,state = final_state, param = param, path_back = path_back_final)
                        self.final_node_list.append(final_node)
                        self.tree.add_node(final_node)
                        
                        # self.assign_shortest_path(tree)
                    # new_goal, path_to_goal, final_cost = self.dubins_steer(new_point, self.goal)
                    # if path_to_goal:
                    #     final_node = Tree_Node(parent = nearest_node, cost = final_cost, state = new_goal)
                    #     tree.add_node(final_node)
                    #     self.final_node_list.append(final_node)
                    #     self.assign_shortest_path(tree)
                random_node = tree.get_nodes()[random.randint(0, len(tree.get_nodes())-1)]
                rewire_node = new_node if random.random() < 0.8 else random_node
                rewired = self.rewiring(new_node, eval=eval, obstacle_check = obstacle_check, baseline = baseline)
                self.rewire_count += 1 if rewired else 0
                self.assign_shortest_path()
                # rewire_count += 1 if rewired else 0
                # self._node.get_logger().info(f"Was tree rewired? {rewired}")

    def assign_shortest_path(self, tree = None):
        """ Assign the shortest path to the tree. Only called if there is a change in the final node list """
        if tree is None:
            tree = self.tree
        # self._node.get_logger().info(f"Assigning shortest path")
        
        min_cost = np.inf
        for final_node_candidate in self.final_node_list :
            
            path, cost = tree.find_path(final_node_candidate)
            if self.verbose:
                self._node.get_logger().info(f"Final node candidate: {final_node_candidate.get_state()} with cost {cost}")
            if cost < min_cost:
                min_cost = cost
                self.path = path
                self.final_node = final_node_candidate
        # if self.verbose:
        self._node.get_logger().info(f"best cost: {min_cost}")
    
    def dubins_steer(self, start, end, free_range = True, optimize = False, obstacle_check = False):
        """ Move from start towards end by step_size """
        # self._node.get_logger().info(f"start: {start} end : {end}")
        direction = np.array(end) - np.array(start)
        direction = direction[:2]  # Ignore Z and yaw
        norm = np.linalg.norm(direction)
        if norm == 0:
            return start, False, np.inf, None
        
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
            return start, False, np.inf, None
        else: 
            #now we check for collisions and the shortest path for a range of headings
            path_exist,best_heading, cost, best_param = self.is_path_collision_free(start,new_point, optimize_heading = optimize, obstacle_check = obstacle_check)
            new_point = (new_point[0], new_point[1], new_point[2], best_heading)
            # self._node.get_logger().info(f"returning new point: {new_point}")
            return new_point, path_exist, cost, best_param

    def is_path_collision_free(self, start, end, optimize_heading = True, obstacle_check = False):
        base_heading = end[3]
        # base_heading = start[3]
        step_heading = 180
        if optimize_heading:
            headings =  base_heading + np.linspace(-180, 180, step_heading, endpoint = True) 
        else:
            headings = [base_heading]
        min_cost = np.inf
        best_param = None
        best_heading = base_heading
        path_exist = False
        path_temp = []
        obstacle_list = None
        # if obstacle_check:
        #         closest_obstacle = self.obstacle_tree.query((start[0], start[1],0), k=1)
        #         distance_to_obstacle = closest_obstacle[0]
        #         if distance_to_obstacle < self.turning_radius:
        #             path_exist = False
        #             return False, base_heading, min_cost, best_param
        #         obstacle_list = [self.obstacles[closest_obstacle[1]]]
        for heading in headings:
            wp_1 = Waypoint(start[0], start[1], start[3])
            wp_2 = Waypoint(end[0], end[1], heading)
            path_exist_temp = False
            
            if obstacle_check:
                param = calc_dubins_path(wp_1, wp_2, self.turning_radius, obstacle_list = self.obstacles, obstacle_tree = self.obstacle_tree)
            else:
                param = calc_dubins_path(wp_1, wp_2, self.turning_radius)
            path_exist_temp = (param.seg_final != [0, 0, 0]) # if no path exists this is [0 0 0]
            #this is computation only for visualization
            # path_temp = sample_between_wps(wp_1,wp_2, self.turning_radius, self.step_dubins, self.obstacles)
            # path_exist_temp = len(path_temp) > 2

            if path_exist_temp:
                path_exist = True
                path_cost = self.turning_radius*sum(param.seg_final)

                if path_cost < min_cost:
                    min_cost = path_cost 
                    best_heading = heading 
                    best_param  = param
            # else: 
                # print(f"Path does not exist for heading: {heading}")
        # if path_exist:
            # self._node.get_logger().info(f"Best heading: {best_heading - base_heading}")

        return path_exist, best_heading, min_cost, best_param
    
    def rewiring(self, new_node, tree = None, existing_node = False, eval = True, obstacle_check = False, baseline = False):
        """ Rewire the tree to reduce cost """
        if tree is None:
            tree = self.tree
        rewired = False
        # rewiring_distance = self.rewiring_distance
        rewiring_distance = self.stepsize*2
        gamma = 2.0
        n_nodes = len(tree.get_nodes())
        d = 2 #dimensions
        rewiring_distance = rewiring_distance* gamma * (np.log(n_nodes) / n_nodes) ** (1 / d)
        # self._node.get_logger().info(f"Rewiring distance: {rewiring_distance}")
        node_list = tree.find_nearest_neighbors(new_node.get_state(), k=self.rewiring_limit, radius=rewiring_distance)
        for iter,node in enumerate(node_list) :
            if iter == self.rewiring_limit:
                break
            valid_node = node != new_node 
            # and node != tree.get_root()
            distance = np.linalg.norm(np.array(new_node.get_state())[0:2] - np.array(node.get_state())[0:2])
            if valid_node :
                # and distance < rewiring_distance:
                path_exist,best_heading,new_cost_segment, param = self.is_path_collision_free(node.get_state(), new_node.get_state(), obstacle_check= obstacle_check)  
                new_state = new_node.get_state()
                new_state = (new_state[0],new_state[1],new_state[2],best_heading)
                new_node.assign_state(new_state)
                path_back = []
                if baseline:
                    path_back_exist = True
                else: 
                    path_back = self.find_reverse_path(new_state, tree, optimize=self.optimize_path_back, obstacle_check = obstacle_check)
                    path_back_exist = len(path_back) > 0

                if  path_exist and path_back_exist: #and (distance < self.stepsize)
                    #compute cost of new path and old path
                    # self._node.get_logger().info(f"rewiring node check")
                    old_path,old_cost = tree.find_path(new_node)

                    # node.assign_parent(new_node)
                    new_path,cost_till_new = tree.find_path(node)
                    new_cost = new_cost_segment + cost_till_new
                    #compare costs
                    if new_cost < old_cost:
                        # rewired = True
                        old_parent = new_node.get_parent()
                        # old_parent.remove_child(new_node)
                        if old_parent is not None:
                            children = old_parent.get_children()
                            children.remove(new_node)
                            old_parent.assign_children(children)
                        new_node.assign_parent(node)
                        new_node.assign_return_path(path_back)
                        node.add_child(new_node)
                        new_node.assign_cost(new_cost_segment)
                        new_node.assign_param(param)
                        continue
                    # else:
                    #     node.assign_parent(old_path[-2])
        iter = 0
        node_list = tree.find_nearest_neighbors(new_node.get_state())
        for iter,node in enumerate(node_list) :
            if iter == self.rewiring_limit:
                break
            valid_node = node != new_node and node != tree.get_root()
            distance = np.linalg.norm(np.array(new_node.get_state())[0:2] - np.array(node.get_state())[0:2])
            if valid_node and distance < rewiring_distance:
                path_exist,_,new_cost_segment, param = self.is_path_collision_free(new_node.get_state(), node.get_state(), optimize_heading=False, obstacle_check= obstacle_check)  

                if  path_exist: #and (distance < self.stepsize)
                    #compute cost of new path and old path
                    # self._node.get_logger().info(f"rewiring node check")
                    old_path,old_cost = tree.find_path(node)
                    if eval : 
                        old_uncertainty_cost = self.compute_path_uncertainty(old_path) 
                        old_cost_adjusted = old_cost - old_uncertainty_cost
                    else: 
                        old_cost_adjusted = old_cost
                    # node.assign_parent(new_node)
                    new_path,cost_till_new = tree.find_path(new_node)
                    new_cost = new_cost_segment +  cost_till_new  
                    if eval: 
                        new_uncertainty_cost = self.compute_path_uncertainty(new_path)
                        new_cost_adjusted = new_cost - new_uncertainty_cost
                    else: 
                        new_cost_adjusted = new_cost

                    if self.verbose:
                        # print(f"\n[REWIRING CHECK] Node ID: {node.get_id()}")
                        print(f"  - Old total cost: {old_cost:.4f}")
                        print(f"  - Old uncertainty cost: {old_uncertainty_cost:.4f}")
                        print(f"  - Adjusted old cost: {old_cost_adjusted:.4f}")
                        print(f"  - New path segment cost: {new_cost:.4f}")
                        print(f"  - New uncertainty cost: {new_uncertainty_cost:.4f}")
                        print(f"  - New total cost: {new_cost_adjusted:.4f}")
                        #this is not being recorded but it probably should.........
                        if new_cost_adjusted < old_cost_adjusted:
                            print("  → REWIRING: new path is better after uncertainty adjustment")
                            ...
                        else:
                            print("  ✗ No rewiring: old path remains better")
                    #compare costs
                    if new_cost_adjusted < old_cost_adjusted:
                        rewired = True
                        old_parent = node.get_parent()
                        # old_parent.remove_child(node)
                        if old_parent is not None:
                            children = old_parent.get_children()
                            children.remove(node)
                            old_parent.assign_children(children )
                        node.assign_parent(new_node)
                        new_node.add_child(node)
                        node.assign_cost(new_cost_segment)
                        node.assign_param(param)
                        continue
                    # else:
                    #     node.assign_parent(old_path[-2])
        return rewired

    def compute_path_cost(self, path):
        """Compute cumulative cost along a path"""
        cost_sum = 0.0

        arc_length = 0.0
        for iter,node in enumerate(path[1:]):
            cost = node.get_cost()
            param = node.get_param()
            # print(param.type)
            if param.type == 'TurnType.LRL' or param.type == 'TurnType.RLR':
                arc_length += (param.seg_final[0] + param.seg_final[1] + param.seg_final[2])
            else :
                arc_length += (param.seg_final[0] + param.seg_final[2]) 
            # cost_sum += (param.seg_final[0] + param.seg_final[1] + param.seg_final[2]) 
            cost_sum += cost

        # cost_sum = cost_sum*self.turning_radius
        arc_length = arc_length*self.turning_radius
        # self._node.get_logger().info(f"cost : {cost_sum}  arc length : {arc_length}") 
        return cost_sum, arc_length
    
    def synthetic_uncertainty(self, x, y):
        cx1,cy1 = 30, -20
        cx2,cy2 = 50, 20
        sigma = 10
        return 5 * np.exp(-((x - cx1) ** 2 + (y - cy1) ** 2) / (2 * sigma ** 2)) + 5 * np.exp(-((x - cx2) ** 2 + (y - cy2) ** 2) / (2 * sigma ** 2))

    def compute_path_uncertainty(self, path):
        """Compute cumulative GP variance (uncertainty) along a path"""
        uncertainty_sum = 0.0
        for node in path:
            x,y = node.get_state()[0:2]
            mean, std_dev = self.trainer.cost_at_sample(x, y)

            #test case time

            # if y< -5:
            #     std_dev = 1
            std_dev = self.synthetic_uncertainty(x,y)
            uncertainty_sum += std_dev**2  # Variance, not standard deviation
        return uncertainty_sum*self.scale_uncertainty

    def simulate_path(self, parent_node, child_node):
        """Helper to simulate a fake path if rewired"""
        simulated_path = []
        node = child_node
        while node is not None and node != parent_node:
            simulated_path.append(node)
            node = node.get_parent()
        simulated_path.append(parent_node)
        return list(reversed(simulated_path))

    def goal_biased_sampling(self, passage_bias = False):
        """ Biased sampling towards the goal """
        sample = random.uniform(0, 1)
        if sample < 0.0001:
            return self.goal
        if passage_bias :
          if sample > 0.0001 and sample < 0.1 : 
            return (random.uniform(self.bounds[0][0], self.bounds[0][1]), 0, 0, 0)
        # elif sample > 0.0001 and sample < 0.3 : 
        # else:
        #     return (random.uniform(self.bounds[0][0], self.bounds[0][1]), 0, 0, 0)
        # return (random.uniform(0, 100), random.uniform(-50, 50), 0, random.uniform(-180, 180)) #random.uniform(-np.pi/6, np.pi/6))
        # return (random.uniform(self.bounds[0]), random.uniform(-50, 50), 0, random.uniform(-180, 180)) #random.uniform(-np.pi/6, np.pi/6))
        return (random.uniform(self.bounds[0][0], self.bounds[0][1]), random.uniform(self.bounds[1][0], self.bounds[1][1]), 0, random.uniform(-180, 180))
    
    def informative_sampling(self, start, goal, cmax= np.inf):
        """ Biased sampling towards the goal """
        if cmax < np.inf:
            cmin = np.linalg.norm((np.array(start)[0:2] - np.array(goal)[0:2]))
            #2 dimensions on our case. heading will be decided by optimizer anyway
            
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
            # # now we plot the elipse
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
    
    def reconnect_to_tree(self, state, obstacle_check = False):
        """Reconnects the current measured state to the tree and sets it as the new root."""
        # Step 1: Find the nearest node in the tree, maybe it should be path not tree akshually
        # nearest_node = self.tree.find_nearest_neighbor(state)
        state = (state[0], state[1], state[2], state[3])
        self._node.get_logger().info(f"current state:{state}")
        final_wp = False
        nearest_node_list = sorted(self.path, key=lambda node: np.linalg.norm(np.array(node.get_state())[0:2] - np.array(state)[0:2]))
        self._node.get_logger().info(f"nearest node list length:{len(nearest_node_list)}")
        curr_wp = -1
        nearest_node = Tree_Node()
        path_exists = False
        for curr_wp,nearest_node_temp in enumerate(nearest_node_list) : 
            if curr_wp <= self.wp_count : 
                    path_exists = False
                    continue
            else :
                # Step 2: Try a direct Dubins connection to the nearest node
                path_exists,_,_,_ = self.is_path_collision_free(state, nearest_node_temp.get_state(), optimize_heading=False, obstacle_check= obstacle_check)
            if path_exists:
                self.wp_count = curr_wp
                nearest_node = nearest_node_temp
                break

        if path_exists:
            self._node.get_logger().info("path to tree exists")
            self._node.get_logger().info(f"nearest node is : {nearest_node.get_state()}")
            self._node.get_logger().info(f"current state is : {state}")
            #forward tree upto the nearest node
            #testing forward tree
            self._node.get_logger().info(f"nearest neighbor children are : {len(nearest_node.get_children())}")
            forward_tree = copy_tree(nearest_node)
            # Direct connection is possible; create a new root node
            new_root = Tree_Node(state=state, parent=None)
            old_root = forward_tree.get_root()
            old_root.assign_parent(new_root)
            # Update the tree structure
            forward_tree.add_node(new_root)
            forward_tree.set_root(new_root)
            # final_node = self.path[-1]
            final_node = self.final_node
            
            for node in forward_tree.get_nodes():
                if node.get_state()[0:2] == final_node.get_state()[0:2]:
                    final_tree_node = node
                    break
            # final_tree_node = forward_tree.find_nearest_neighbor(final_node.get_state())
            self._node.get_logger().info(f"final node is : {final_tree_node.get_state()}")
            path_forward ,cost= forward_tree.find_path(final_tree_node)

            if len(path_forward) == 2:
                #path length being 2 means that the goal is the immediate next waypoint
                final_wp = True
            dubins_input_forward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_forward 
                    if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None]
            dubins_out_forward, original_indices = sample_complete_plan(dubins_input_forward, self.turning_radius, self.step_dubins, self.obstacles, self.obstacle_tree)
            self.original_wp_indices = [int(i) for i in original_indices]
            #backward path
            backward_tree = copy_tree(self.path[0], final_node = self.path[curr_wp-2])
            path_backward = self.find_reverse_path(state, tree = backward_tree, optimize=True, obstacle_check = obstacle_check)
            self._node.get_logger().info(f"path length: {len(path_backward)}")
            dubins_input_backward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_backward
                if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None] 
            dubins_out_backward, _ = sample_complete_plan(dubins_input_backward, self.turning_radius, self.step_dubins, self.obstacles, self.obstacle_tree)
            self.visualize_tree(np.array(dubins_out_forward),np.array(dubins_out_backward), path = path_forward, tree = forward_tree)
            # self._node.get_logger().info(f"the original indices : {original_indices}")
            # convert back to pose2D for transfer
            self.original_wp_indices = [int(i) for i in original_indices]
            # give waypoints till the first original index to fawllow

            for wp_index,wp in enumerate(dubins_out_forward):
                #here we use the goal tolerance from the action client
                if np.linalg.norm(np.array(wp[0:2]) - np.array(state[0:2])) > 7.0:
                    # self._node.get_logger().info(f"found start waypoint {wp}")
                    break
            # dubins_first_waypoint = dubins_out_forward[0:self.original_wp_indices[1]+1]
            dubins_first_waypoint = dubins_out_forward[wp_index:self.original_wp_indices[1]+1]

            goal_msg = self.send_waypoints(dubins_first_waypoint, path_forward)
            self._node.get_logger().info(f"first waypoint : {dubins_first_waypoint[0]}")
            self._node.get_logger().info(f"distance from start to first waypoint : {np.linalg.norm(np.array(dubins_first_waypoint[0][0:2]) - np.array(state[0:2]))}")
            # self._node.get_logger().info(f"current final wp : {self._ac.waypoint_queue[-1]}")
            
            return goal_msg, final_wp
            # return True  # Successfully reconnected 
        else: 
            self._node.get_logger().info("Path from current position to the nearest node doesn't exist")
            #what do i give back here 
            return [], final_wp

        
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

    def find_reverse_path(self, state, tree = None, optimize = False, obstacle_check = False):
        """ Returns the path from the root to the end node."""
        if tree is None:
            tree = self.tree
        path = []
        alt_path = True
        optimize_heading = optimize
        
        # if optimize:
        #     self._node.get_logger().info(f"Optimizing reverse path. state : {state} ")
        nearest_node_list = tree.find_nearest_neighbors(state) #THIS IS A GOOD BACKUP THO
        # if len(nearest_node_list) == 0:
        #     nearest_node_list = tree.find_nearest_neighbors(state)
        #     alt_path = True
        if len(nearest_node_list) > 10 :
            nearest_node_list = nearest_node_list[0:10]
        # self._node.get_logger().info(f"length of nearest neighbors: {len(nearest_node_list)}")
        reverse_path_dict = {} 
        for nearest_node in nearest_node_list:
            
            reverse_path = []
            total_cost = 0


            if alt_path:
                path, cost = tree.find_path(nearest_node) 
            else: 
                index = self.path.index(nearest_node)
                path  = self.path[0:index+1]
            # if optimize : 
            #     path = self.path
            # path = self.path
            # self._node.get_logger().info(f"Root of tree : {tree.get_root().get_state()}")
            # self._node.get_logger().info(f"Path length till neighbor : {len(path)}")
            nearest_node_state = nearest_node.get_state()
            # if nearest_node_state[0:2] == state[0:2]:
            
                # continue
            nearest_node_reverse_state = nearest_node_state[0], nearest_node_state[1], nearest_node_state[2], nearest_node_state[3] + 180
            first_point,first_path_exist, first_cost, param = self.dubins_steer(state, nearest_node_reverse_state, free_range = True, optimize=optimize_heading, obstacle_check = obstacle_check)  # Steer towards the parent
            total_cost+= first_cost
            # new_parent = Tree_Node(state=state, parent=None)
            #right now it gives the first found path, for speeding up forward computation, but we can optimize when we actually try to return?
            if not first_path_exist:
                continue
            
            first_node = Tree_Node(state = state, parent = None)
            reversed_node = Tree_Node(state = first_point,cost = first_cost, parent = first_node, param = param)
            reverse_path.append(first_node)
            reverse_path.append(reversed_node)
            new_parent = reversed_node
            # self._node.get_logger().info("first segment is steerable")
            # if not optimize:
            #     path.append(Tree_Node(state=state, parent=nearest_node))

            # Go through the original path in reverse order
            for i, p in enumerate(reversed(path[0:-1])):
                # if i == 0 :
                #     continue
                # print(i)

                reversed_state = (p.get_state()[0], p.get_state()[1], p.get_state()[2], p.get_state()[3] + 180)
                parent_state = new_parent.get_state()
                #THIS IS *probably* TRIVIAL, IT IS TRAVERSIBLE THAT'S WHY IT EXISTS IN THE TREE AT ALL
                new_point, path_exist, seg_cost, param = self.dubins_steer(parent_state, reversed_state, optimize = optimize_heading, obstacle_check = obstacle_check)  # Steer towards the parent
                
                if not path_exist:
                    # if optimize:
                    #     self._node.get_logger().info(f"States {parent_state} and {reversed_state}")
                    #     self._node.get_logger().info(f"segment {i} is not steerable")
                    reverse_path.clear()
                    break
                # if optimize:
                arc_segment = 0
                if param is not None:
                    arc_segment = (param.seg_final[0] + param.seg_final[1] + param.seg_final[2]) * self.turning_radius if param.type == 'TurnType.LRL' or param.type == 'TurnType.RLR' else (param.seg_final[0] + param.seg_final[2]) * self.turning_radius
                # path_length_remaining, remaining_arc = self.compute_path_cost(path[0:len(path) - i - 1])
                seg_cost = p.get_cost()
                total_cost += seg_cost + arc_segment*self.SCALE_ARCLENGTH
                reversed_node = Tree_Node(state=new_point, cost=seg_cost, parent=new_parent, param = param)
                reverse_path.append(reversed_node)
                new_parent = reversed_node

                # reverse_path = [Tree_Node(state=(p.get_state()[0], p.get_state()[1], p.get_state()[2], -p.get_state()[3]), parent=path[i+1] if i+1 < len(path) else None) for i, p in enumerate(reversed(path))]
            
            if len(reverse_path) > 1:
                # self._node.get_logger().info(f"Reverse path length: {len(reverse_path)}")
                if not optimize : 
                    return reverse_path
                else:
                    reverse_path_dict[total_cost] = reverse_path
                    # self._node.get_logger().info(f"Reverse path cost: {total_cost} and length: {len(reverse_path)}")
        # self._node.get_logger().info("No reverse path found")
        # self._node.get_logger().info(f"length of dict: {len(reverse_path_dict)}")
        if optimize:
            if reverse_path_dict == {}:
                # self._node.get_logger().info("No reverse path found(optimize)")
                # rand_point = self.goal_biased_sampling()
                
                # new_point, path_exist, cost, param = self.dubins_steer(state, rand_point, free_range = False, obstacle_check = obstacle_check )  # Steer towards the random point
                # return self.dyn_rrt(state, tree, optimize = optimize, obstacle_check=obstacle_check)
                return []
            best_cost = min(reverse_path_dict.keys())
            # self._node.get_logger().info(f"Best reverse path cost: {best_cost}")
            best_path = reverse_path_dict[best_cost]
            # self._node.get_logger().info(f"Final node: {self.final_node.get_state()}")
            # for wp in best_path:
                # self._node.get_logger().info(f"Reverse path waypoint: {wp.get_state()}")
            return best_path
        # else:
            # reverse_path.clear()
        return []

    def dyn_rrt(self, state,  tree, optimize = True, obstacle_check = False):
        """connect to old tree and rewire it with the new tree"""
        #Step 1 : Initialize new tree
        self._node.get_logger().info(f"DYNAMIC RRT STARTED state: {state}")
        root_node = Tree_Node(state = state)
        new_tree = Tree(root_node, visualize=False)
        # check_tree = False
        optimize_heading = optimize
        for  iter in range(10000):
            # if random.uniform(0, 1) < 0.8:
            #     # Step 2: Sample a point in the environment
            rand_point = self.goal_biased_sampling()
            # else:
            #     rand_point = self.informative_sampling(state, self.start)

            # Step 3: Find the nearest node in the tree
            nearest_node = new_tree.find_nearest_neighbor(rand_point)

            # Step 4: Steer towards the random point
            new_point, path_exist, cost, param = self.dubins_steer(nearest_node.get_state(), rand_point, free_range = False, obstacle_check = True)  # Steer towards the random point
            if path_exist:
                # Step 5: Create a new node and add it to the tree
                new_node = Tree_Node(state=new_point, cost=cost, parent=nearest_node, param=param)
                new_tree.add_node(new_node)
                nearest_node.add_child(new_node)

                # Step 6: Check for connection to old tree
                
                reverse_path_dict = {}
                nearest_old_tree_nodes = tree.find_nearest_neighbors(new_point)
                for nearest_old_tree_node in nearest_old_tree_nodes:
                    reverse_path = []
                    total_cost = 0
                    path, cost = tree.find_path(nearest_old_tree_node) 
                    path_exists, _, _, _ = self.is_path_collision_free(new_point, nearest_old_tree_node.get_state(), obstacle_check=True)
                    if path_exists:
                        # Step 7: Reconnect to the old tree
                        nearest_old_node_state = nearest_old_tree_node.get_state()
                        nearest_node_reverse_state = nearest_old_node_state[0], nearest_old_node_state[1], nearest_old_node_state[2], nearest_old_node_state[3] + 180
                        first_point,first_path_exist, first_cost, param = self.dubins_steer(state, nearest_node_reverse_state, free_range = True, optimize=optimize_heading, obstacle_check = obstacle_check)  # Steer towards the parent
                        total_cost+= first_cost
                        # new_parent = Tree_Node(state=state, parent=None)
                        #right now it gives the first found path, for speeding up forward computation, but we can optimize when we actually try to return?
                        if not first_path_exist:
                            self._node.get_logger().info("first segment is not steerable")
                            self._node.get_logger().info(f"State: {state} and nearest node state: {nearest_node_reverse_state}")
                            continue
                        
                        first_node = Tree_Node(state = state, parent = None)
                        reversed_node = Tree_Node(state = first_point,cost = first_cost, parent = first_node, param = param)
                        reverse_path.append(first_node)
                        reverse_path.append(reversed_node)
                        new_parent = reversed_node
                        # self._node.get_logger().info("first segment is steerable")
                        # if not optimize:
                        #     path.append(Tree_Node(state=state, parent=nearest_node))

                        # Go through the original path in reverse order
                        for i, p in enumerate(reversed(path[0:-1])):
                            # if i == 0 :
                            #     continue
                            # print(i)

                            reversed_state = (p.get_state()[0], p.get_state()[1], p.get_state()[2], p.get_state()[3] + 180)
                            parent_state = new_parent.get_state()
                            #THIS IS *probably* TRIVIAL, IT IS TRAVERSIBLE THAT'S WHY IT EXISTS IN THE TREE AT ALL
                            new_point, path_exist, seg_cost, param = self.dubins_steer(parent_state, reversed_state, optimize = optimize_heading, obstacle_check = obstacle_check)  # Steer towards the parent
                            
                            if not path_exist:
                                if optimize:
                                    self._node.get_logger().info(f"States {parent_state} and {reversed_state}")
                                    self._node.get_logger().info(f"segment {i} is not steerable")
                                reverse_path.clear()
                                break
                            # if optimize:
                            arc_segment = 0
                            if param is not None:
                                arc_segment = (param.seg_final[0] + param.seg_final[1] + param.seg_final[2]) * self.turning_radius if param.type == 'TurnType.LRL' or param.type == 'TurnType.RLR' else (param.seg_final[0] + param.seg_final[2]) * self.turning_radius
                            # path_length_remaining, remaining_arc = self.compute_path_cost(path[0:len(path) - i - 1])
                            seg_cost = p.get_cost()
                            total_cost += seg_cost + arc_segment*self.SCALE_ARCLENGTH
                            reversed_node = Tree_Node(state=new_point, cost=seg_cost, parent=new_parent, param = param)
                            reverse_path.append(reversed_node)
                            new_parent = reversed_node

                            # reverse_path = [Tree_Node(state=(p.get_state()[0], p.get_state()[1], p.get_state()[2], -p.get_state()[3]), parent=path[i+1] if i+1 < len(path) else None) for i, p in enumerate(reversed(path))]
                        
                        if len(reverse_path) > 1:
                            # self._node.get_logger().info(f"Reverse path length: {len(reverse_path)}")
                            if not optimize : 
                                return reverse_path
                            else:
                                reverse_path_dict[total_cost] = reverse_path
                                # self._node.get_logger().info(f"Reverse path cost: {total_cost} and length: {len(reverse_path)}")
                    # self._node.get_logger().info("No reverse path found")
                    # self._node.get_logger().info(f"length of dict: {len(reverse_path_dict)}")
                    if optimize:
                        if reverse_path_dict == {}:
                            # self._node.get_logger().info("No reverse path found(optimize)")
                            # rand_point = self.goal_biased_sampling()
                            
                            # new_point, path_exist, cost, param = self.dubins_steer(state, rand_point, free_range = False, obstacle_check = obstacle_check )  # Steer towards the random point
                            # return self.find_reverse_path(state, tree, optimize = optimize, obstacle_check=obstacle_check)
                            continue
                        best_cost = min(reverse_path_dict.keys())
                        self._node.get_logger().info(f"Best reverse path cost: {best_cost}")
                        best_path = reverse_path_dict[best_cost]
                        # self._node.get_logger().info(f"Final node: {self.final_node.get_state()}")
                        # for wp in best_path:
                            # self._node.get_logger().info(f"Reverse path waypoint: {wp.get_state()}")
                        return best_path
        return []        
        
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
        
    def visualize_tree(self, waypoint_array_forward, waypoint_array_backward, path, tree = None, visualize_back = True, xx =None, yy = None, pred_mean = None, pred_std = None):

        """ Plot the waypoints and obstacles """
        plt.figure(figsize=(8, 8))
        if xx is not None:
            # Plot the GP uncertainty map first (so it's in the background)
            std_plot = plt.contourf(xx.numpy(), yy.numpy(), pred_std.numpy(), levels=20, cmap="viridis", alpha=0.5)
            plt.colorbar(std_plot, label="Predictive Stddev (Uncertainty)")
        if tree is None:
            tree = self.tree
        # Plot waypoints
        if len(waypoint_array_forward) > 0:
            for i in self.original_wp_indices:
                plt.scatter(waypoint_array_forward[i][0], waypoint_array_forward[i][1], color='g', marker='x')
                # , label="Original Waypoint" if 'Original Waypoint' not in plt.gca().get_legend_handles_labels()[1] else "")
            x, y = waypoint_array_forward[:, 0], waypoint_array_forward[:, 1]
            plt.plot(x, y, marker=',', linestyle='-', color='b', label="Path")
            # plt.scatter(x, y, color='r', label="Waypoints")  # Highlight waypoints
        if len(waypoint_array_backward) > 0 and visualize_back:
            x, y = waypoint_array_backward[:, 0], waypoint_array_backward[:, 1]
            plt.plot(x, y, marker=',', linestyle='-', color='r', label  = "Reverse Path")
        #     plt.scatter(x, y, color='r')
            
        # Plot obstacles with their radii
        for ox, oy, _, r in self.obstacles:  # Ignoring Z
            obstacle_circle = plt.Circle((ox, oy), r, color='gray', alpha=0.5, fill=True)
            plt.gca().add_patch(obstacle_circle)
            plt.scatter(ox, oy, color='k', marker='x', label="Obstacle" if 'Obstacle' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        for node in tree.get_nodes():
            if node.get_parent() is not None: 
            #and node not in path:
                parent = node.get_parent()
                node_state = node.get_state()
                parent_state = parent.get_state()
                node_wp = Waypoint(node_state[0], node_state[1], node_state[3])
                parent_wp = Waypoint(parent_state[0], parent_state[1], parent_state[3])
                path = [(parent_wp.x,parent_wp.y,parent_wp.psi)]
                path_append = sample_between_wps(parent_wp, node_wp, self.turning_radius, 1.0, self.obstacles, self.obstacle_tree)
                path.extend(path_append)
                path.append((node_wp.x,node_wp.y,node_wp.psi))
                edge_array = np.array(path)
                edge_x, edge_y = edge_array[:, 0], edge_array[:, 1]
                plt.plot(edge_x, edge_y, marker=',', linestyle=':', color='c')
                plt.scatter(node_wp.x, node_wp.y, color='g', marker='x')

                # plt.plot([node.get_state()[0], parent.get_state()[0]], [node.get_state()[1], parent.get_state()[1]], color='g', linestyle='-', linewidth=0.5)
        # Start and Goal positions
        start_x, start_y = self.start[:2]  # Ignore Z
        goal_x, goal_y = self.goal[:2]  # Ignore Z
        # Plot start and goal
        root = tree.get_root()
        root_state = root.get_state()
        plt.scatter(root_state[0],root_state[1], color='hotpink', marker='s', s=150, label="Root")
        plt.scatter(start_x, start_y, color='g', marker='s', s=150, label="Start")  # Green Square
        plt.scatter(goal_x, goal_y, color='y', marker='*', s=200, label="Goal")  # Yellow Star

        # std_plot = plt.contourf(xx.numpy(), yy.numpy(), pred_std.numpy(), levels=20, cmap="magma")
        # # axs[2].set_title("GP Predictive Stddev (Uncertainty)")
        # plt.figure.colorbar(std_plot)
        # x = np.linspace(0, 100, 100)
        # y = np.linspace(-50, 50, 100)
        # xx, yy = np.meshgrid(x, y)
        # pred_std = self.synthetic_uncertainty(xx, yy)
        # std_plot = plt.contourf(xx,yy, pred_std, levels=20, cmap="viridis", alpha=0.5)
        # plt.colorbar(std_plot)
        # Plot settings
        plt.xlim(0, 100)
        plt.ylim(-50, 50)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Waypoint Path with Obstacles")
        plt.legend()
        plt.grid()
        plt.axis("equal")  # Ensures equal scaling for X and Y
        plt.show()

    def trainGP(self):
        """train the surrogate model of the GP"""
        # self.trainer.plot_training_data()
        self.trainer.train()
        # xx, yy, pred_mean, pred_std = self.trainer.predict()
        # self.trainer.plot_results(xx, yy, pred_mean, pred_std)
        
    def eval_manager(self, obstacles = False, info = False):
        """ Evaluate the planner """
        start = None
        #get initial orientation
        if not self.obstacles_populated: #will temporarily stay here
            self.get_obstacles_from_tf()
            if not obstacles:
                self.obstacles_populated = True
            if not self.obstacles_populated:
                return
        if not self.get_pose()[0]:
            self._node.get_logger().info("Waiting for initial pose")
            return

        else:
            if self.start[3] is None:
                self.start = (self.get_pose()[0], self.get_pose()[1], 0, self.get_pose()[2]) 
                self._node.get_logger().info(f"Initial pose: {self.start}")
                start = self.start
                metrics = []
                #evaluate length and comp time for a range of goals
                n_trials = 50
                for trial in range(n_trials):
                    # self._node.get_logger().info("Train GP Model")
                    #train
                    # self.trainGP()
                    self._node.get_logger().info(f"Trial {trial + 1}/{n_trials}")
                    self.initialize_tree(self.start)
                    #maybe RRT here
                    # self._node.get_logger().info("Running RRT")
                    #first wp
                    # self._node.get_logger().info("Sending Goal")
                    goal = (random.uniform(0, 100), random.uniform(-50, 50), 0, random.uniform(-180, 180))
                    self.goal = goal
                    start_time  = time.time()
                    self.goal_msg, final_waypoint_bool = self.run_rrt(start,goal)
                    end_time = time.time()
                    path_length, arc_length = self.compute_path_cost(self.path)
                    curvature_percentage = arc_length/path_length*100
                    compute_time = end_time - start_time
                    if info:
                        info_cost = self.compute_path_uncertainty(self.path)
                    nodes_sampled = len(self.tree.get_nodes())
                    euclidean = np.linalg.norm(np.array(self.start[0:2]) - np.array(goal[0:2]))
                    if info:
                        metrics.append({
                            "path_length": path_length,
                            "euclidean": euclidean,
                            "optimality": path_length / euclidean,
                            "computation_time": compute_time,
                            "nodes_sampled": nodes_sampled,
                            "info_cost": info_cost,
                            "curvature" : curvature_percentage
                        })
                    else:
                        metrics.append({
                            "path_length": path_length,
                            "euclidean": euclidean,
                            "optimality": path_length / euclidean,
                            "computation_time": compute_time,
                            "nodes_sampled": nodes_sampled,
                            "curvature" : curvature_percentage
                        })
                #write to a csv file
                metrics_df = pd.DataFrame(metrics)
                metrics_df.to_csv("metrics.csv", index=False)
                self._node.get_logger().info("Metrics saved to metrics.csv")
                self.timer_eval.cancel()
                # self._node.get_logger().info("Ending Timer")

    def online_manager(self):
        """ Periodically check for obstacles and run RRT """
        start = None
        obstacle_check = False
        self.bounds = ((0,60),(-30,30))
        goal = (48,0,0,0)
        self.goal = goal
        #get initial orientation
        if not self.obstacles_populated: #will temporarily stay here
            self.get_obstacles_from_tf()
            # self.obstacles_populated = True
            return
        
        if not self.get_pose()[0]:
            self._node.get_logger().info("Waiting for initial pose")
            return

        # if self.start[3] is None:
        #         self.start = (self.get_pose()[0], self.get_pose()[1], 0, self.get_pose()[2])
        # lambda_values = [0.1, 0.5, 1.0, 2.0]
        # # lambda_values = [0.1, 0.5]
        # stat_table = {}
        # for lambda_value in lambda_values:
        #     self.LAMBDA = lambda_value
        #     self._node.get_logger().info(f"Running evaluation with lambda: {lambda_value}")
        #     stat_table[lambda_value] = self.run_evaluation_loop(n_trials=5, use_gp=True, visualize=False)
        # self.plot_total_cost_vs_lambda(stat_table)
        # self.plot_path_and_uncertainty_vs_lambda(stat_table)

        # self.timer.cancel()
        
        else:
            if self.start[3] is None:
                self.start = (self.get_pose()[0], self.get_pose()[1], 0, self.get_pose()[2]) 
                self._node.get_logger().info(f"Initial pose: {self.start}")
                start = self.start
                # self._node.get_logger().info("Train GP Model")
                #train
                # self.trainGP()
                self.initialize_tree(self.start)
                #maybe RRT here
                self._node.get_logger().info("Running RRT")
                #first wp
                # self._node.get_logger().info("Sending Goal")

                self.goal_msg, final_waypoint_bool = self.run_rrt(start, self.goal, obstacle_check = obstacle_check)
                self._ac.waypoint_queue.extend(self.goal_msg[1:])
                self._ac.send_goal()
                self.goal_sent_flag = True
                self.start_time_wp = time.time()
        
        #this takes care of ending the loop when we reach the final waypoint 
        final_waypoint_bool = False #should be handled in the first rrt call but I keep this here for now
       
        # if self.path != [] :
        if not final_waypoint_bool :
            #now we run rrt
            if not self.goal_sent_flag : 
                pose_current = self.get_pose()
                start = (pose_current[0], pose_current[1], 0, pose_current[2])
                self._node.get_logger().info(f"pose : {start}")
                # self._node.get_logger().info("Running RRT")
                # self.goal_msg, final_waypoint_bool = self.run_rrt(start, self.goal)
                #maybe instead of running RRT all over again, we find a way back to the tree. read papers???
                self.goal_msg, final_waypoint_bool = self.reconnect_to_tree(start, obstacle_check = obstacle_check )
                #send goal = first waypoint to action client
                self._node.get_logger().info("Sending Goal")
                #here we need to wait for feedback msg from le client
                #once we have waited, new actual state from tf, and update the state so that rrt is run using it next time. 
                # self._node.get_logger().info(f"length of queue {len(self._ac.waypoint_queue)}")
                self._ac.waypoint_queue.extend(self.goal_msg)
                # self._node.get_logger().info(f"first queue element : {self._ac.waypoint_queue[0]}")
                self._node.get_logger().info(f"battery level : {self.battery_state}")
                self._ac.send_goal()
                self.goal_sent_flag = True
                self.start_time_wp = time.time()
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
            # if not self._ac.check_queue_for_waypoint() :
            if self._ac.wps_traversed :
            
                # new_start = self.get_pose()
                # self.start = new_start
                self._node.get_logger().info("all waypoints traversed. setting flag to done")
                self.goal_sent_flag = False
            if self.start_time_wp is not None: 
                end_time = time.time()
                time_elapsed = end_time - self.start_time_wp
                # self._node.get_logger().info(f"Time elapsed: {time_elapsed}")
                if time_elapsed > 60 :
                    self.goal_sent_flag = False
                    self.start_time_wp = end_time
        else : 
            #cancel timer
            self._node.get_logger().info("Ending Timer")
            self.timer.cancel()
        # else : 
        #     self._node.get_logger().info("Tree not initialized")
        #     return

    def run_evaluation_loop(self, n_trials=10, use_gp=True, visualize=True):
        """
        Run multiple planning trials with a fixed goal.
        Useful for evaluating different lambda values or planner variants.
        """
        stats = {
            "success": 0,
            "path_lengths": [],
            "uncertainties": [],
            "total_costs": [],
            "failures": [],
        }

        for trial in range(n_trials):
            self._node.get_logger().info(f"Trial {trial + 1}/{n_trials}")
            self.initialize_tree(self.start)

            # # 1. Get fresh start and obstacle data
            # self.get_obstacles_from_tf()
            # start_pose = self.get_pose()
            # if not start_pose[0]:
            #     self._node.get_logger().warn("No pose. Skipping trial.")
            #     stats["failures"].append("no_pose")
            #     continue
            # self.start = (start_pose[0], start_pose[1], 0, start_pose[2])
            # self.initialize_tree(self.start)

            # # 2. Optionally train GP model
            # if use_gp:
            #     self.trainGP()

            # 3. Run RRT and record path
            self._node.get_logger().info("Running RRT...")
            #path is goal_msg akshually
            path, final_wp_reached = self.run_rrt(self.start, self.goal)
            if path is None:
                self._node.get_logger().warn("Failed to reach goal.")
                stats["failures"].append("rrt_fail")
                continue

            # 4. Evaluate results
            path_length, arc_length = self.compute_path_cost(self.path)
            #if i give self.path here it means we check for the bigger waypoints and not hte dubins ones. If I do path, it will be dubins but Ill have to change logic for posestamped from the node_tree
            uncertainty_cost = self.compute_path_uncertainty(self.path)
            lambda_weight = self.LAMBDA if hasattr(self, 'LAMBDA') else 1.0
            total_cost = path_length + lambda_weight * uncertainty_cost

            stats["success"] += 1
            stats["path_lengths"].append(path_length)
            stats["uncertainties"].append(uncertainty_cost)
            stats["total_costs"].append(total_cost)

            self._node.get_logger().info(f"Path length: {path_length:.2f}, Uncertainty: {uncertainty_cost:.2f}, Total: {total_cost:.2f}")

            # 5. Optional visualization
            if visualize:
                self.visualize_tree(waypoint_array_forward=np.array(path), waypoint_array_backward=[], path=path)

        # 6. Summary
        self._node.get_logger().info("Evaluation Done")
        self._node.get_logger().info(f"Success rate: {stats['success']}/{n_trials}")
        return stats

    def scenario_manager(self):
        """ Evaluate the planner """
        start = None
        baseline = False
        obstacle_check = True
        #get initial orientation
        if not self.obstacles_populated: #will temporarily stay here
            self.get_obstacles_from_tf()
            
            if not self.obstacles_populated:
                return
        if not self.get_pose()[0]:
            self._node.get_logger().info("Waiting for initial pose")
            return

        else:
            if self.start[3] is None:
                self.start = (self.get_pose()[0], self.get_pose()[1], 0, self.get_pose()[2]) 
                self._node.get_logger().info(f"Initial pose: {self.start}")
                start = self.start
                metrics_baseline= []
                metrics_safe = []
                #evaluate length and comp time for a range of goals
                n_trials = 500
                for trial in range(n_trials):
                    # self._node.get_logger().info("Train GP Model")
                    #train
                    # self.trainGP()
                    self._node.get_logger().info(f"Trial {trial + 1}/{n_trials}")
                    self.initialize_tree(self.start)

                    if baseline:
                        passage_bias = True
                    else: 
                        passage_bias = False

                        
                    #maybe RRT here
                    # self._node.get_logger().info("Running RRT")
                    #first wp
                    # self._node.get_logger().info("Sending Goal")
                    self.bounds = ((0,60),(-30,30))
                    goal = (48,0,0,0)
                    self.goal = goal
                    # goal = (random.uniform(0, 100), random.uniform(-50, 50), 0, random.uniform(-180, 180))
                    # self.goal = goal
                    start_time  = time.time()
                    
                    # self.goal_msg, final_waypoint_bool = self.run_rrt(start,goal, obstacle_check=True)
                    
                    self.goal_msg, final_waypoint_bool = self.run_rrt(start,goal, obstacle_check=True,baseline=baseline, passage_bias = passage_bias, rewire = False)
                    end_time = time.time()
                    compute_time = end_time - start_time
                    # path_length, arc_length = self.compute_path_cost(self.path)
                    # curvature_percentage = arc_length/path_length*100
                    if len(self.path) == 2:
                        self._node.get_logger().info("Final waypoint reached")
                        continue
                    retreat_index = random.randint(1, len(self.path)-2)
                    retreat_node = self.path[retreat_index]
                    path_length, arc_length = self.compute_path_cost(self.path[0:retreat_index+1])

                    if not baseline: 
                        # path_back_length, arc_length_back = self.compute_path_cost(self.final_node.get_path())
                        back_computation_time = 0.0
                        start_time = time.time()
                        path_back = self.find_reverse_path(retreat_node.get_state(), self.tree, optimize=True, obstacle_check=obstacle_check)
                        if len(path_back) ==0:
                            path_back = self.final_node.get_return_path()
                        end_time = time.time()
                        back_computation_time = end_time - start_time
                        path_back_length, arc_length_back = self.compute_path_cost(path_back)
                        metrics_safe.append({ 
                            "path_length": path_length,
                            "path_length": path_back_length,
                            "computation_time": compute_time,
                            "back_computation_time" : back_computation_time,  
                        })

                    else:
                        old_start = self.start
                        #find waypoint about x% from the start(60)(extract from goal message, use the fact that they're equally spaced)
                        #pick random waypoint from path that and assume we spent 40 perc of the battery getting there
                        
                        self.start = retreat_node.get_state()
                        # start = (32, 0, 0, 0)
                        # start = self.start
                        self.goal = old_start
                        #run rrt* back to start from there
                        self.initialize_tree(self.start)
                        start_time  = time.time()
                        goal_msg, final_waypoint_bool = self.run_rrt(start,goal, obstacle_check=True,baseline=baseline, passage_bias = False, rewire = False)
                        
                        end_time = time.time()
                        back_computation_time = end_time - start_time
                        # get metrics(path length, what else??????)
                        if len(goal_msg) == 0:
                            self._node.get_logger().warn("No path found back to start")
                            path_back_length = np.inf
                            
                        else:
                            path_length_back, arc_length = self.compute_path_cost(self.path)
                        metrics_baseline.append({
                            "path_length": path_length,
                            "path_length_back": path_length_back,
                            "computation_time": compute_time,
                            "back_computation_time" : back_computation_time
                        })
                        self.start = old_start
                        # start = old_start
                        self.goal = goal
                if baseline:
                    metrics_baseline_df = pd.DataFrame(metrics_baseline)
                    metrics_baseline_df.to_csv("metrics_baseline.csv", index=False)
                    self._node.get_logger().info("Metrics saved to metrics_baseline.csv")
                else:
                    metrics_safe_df = pd.DataFrame(metrics_safe)
                    metrics_safe_df.to_csv("metrics_safe.csv", index=False)
                    self._node.get_logger().info("Metrics saved to metrics_safe.csv")
    
    def plot_path_and_uncertainty_vs_lambda(self, stat_table):
        lambdas = sorted(stat_table.keys())
        path_lengths = [np.mean(stat_table[lam]['path_lengths']) for lam in lambdas]
        uncertainties = [np.mean(stat_table[lam]['uncertainties']) for lam in lambdas]

        plt.figure(figsize=(10, 5))
        plt.plot(lambdas, path_lengths, 'bo-', label="Path Length")
        plt.plot(lambdas, uncertainties, 'ro-', label="Uncertainty Cost")
        plt.xlabel("Lambda")
        plt.ylabel("Cost Component Value")
        plt.title("Path Length vs Uncertainty vs Lambda")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_total_cost_vs_lambda(self, stat_table):
        lambdas = sorted(stat_table.keys())
        means = [np.mean(stat_table[lam]['total_costs']) for lam in lambdas]
        stds = [np.std(stat_table[lam]['total_costs']) for lam in lambdas]

        plt.figure(figsize=(8, 5))
        plt.errorbar(lambdas, means, yerr=stds, fmt='o-', capsize=5, label="Total Cost")
        plt.xlabel("Lambda (Weight on Uncertainty)")
        plt.ylabel("Total Cost (Path + λ·Uncertainty)")
        plt.title("Total Cost vs Lambda")
        plt.grid(True)
        plt.legend()
        plt.show()

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