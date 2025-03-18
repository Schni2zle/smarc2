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
from RRTPlanner.Node_tree import Tree_Node, Tree
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
                 goal_tolerance: float = 3.0) -> None:
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
        self.start = (0, 0, 0, 0)  # Start position
        self.goal = (65, 5, 0, 0)  # Goal position
        # self.goal = (20, -40, 0, 0)  # Goal position

        self.rewiring_distance = 2.0  # Rewiring distance
        self.goal_tolerance = goal_tolerance  # Goal tolerance radius
        self.timer = self._node.create_timer(0.1, self.manager)  # Timer to get obstacles periodically
        self.obstacles_populated = False  # Flag to check if obstacles are populated

        self.sam_states = StateInformation(self._node)

        # RRT variables 
        self.forward_tree = None
        self.backward_tree = None

        # RRT parameters
        self.stepsize = 6.0
        self.turning_radius = 3.0
        self.step_dubins = 0.5
        self.original_wp_indices = []
        self.goal_sent = False  




    def get_orientation(self):
        """ Get initial orientation of the robot """
        trans = TransformStamped()
        try:
            trans = self.tf_buffer.lookup_transform('sam_auv_v1/base_link_gt', f'sam_auv_v1/odom_gt', rclpy.time.Time(seconds=0))
            quat = trans.transform.rotation
            roll, pitch, yaw = tf_transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            return yaw*180/np.pi
        except Exception as e:
            self._node.get_logger().info(f"Couldn't lookup transform {e}")
            return None

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
                    self._node.get_logger().info(f"Detected position of obstacle {(x,y)}")
                    self._node.get_logger().info(f"Detected radius of obstacle {radius}")
                    
                except Exception as e:
                    self._node.get_logger().info(f"Couldn't lookup transform {e}")
                    continue  # Ignore missing obstacles
        if len(self.obstacles) == num_obstacles_gt:
            self.obstacles_populated = True
        self._node.get_logger().info(f"Detected {len(self.obstacles)} obstacles")

    def run_rrt(self, start, goal):
        """ Run RRT algorithm to find a collision-free path """
        # path = [self.start]

        forward_root_node = Tree_Node(state = self.start)
        self.visualize = False
        self.forward_tree = Tree(forward_root_node, visualize=self.visualize) 
        rewire_count = 0
        self.sam_states.check_state()
        for _ in range(100000):  # Max iterations 
            self.sam_states.check_state()
            rand_point = self.goal_biased_sampling()
            # self._node.get_logger().info(f"Random point: {rand_point}")
            nearest_node = Tree_Node()
            nearest_node = self.forward_tree.find_nearest_neighbor(rand_point)   # returns node with minimum cost
            path_back = self.find_reverse_path(rand_point, nearest_node)
            path_back_exist = len(path_back) > 0
            # self._node.get_logger().info(f"Nearest node: {nearest_node.get_state()} and random point: {rand_point}")
            new_point, path_exist = self.dubins_steer(nearest_node.get_state(), rand_point)  # Steer towards the random point
            steerable = new_point != nearest_node.get_state()             
            # steerable = new_point != None
            if steerable:
                if path_exist and path_back_exist: # If path exists, add the new node to the tree
                    new_node = Tree_Node(parent = nearest_node, state = new_point)
                    self.forward_tree.add_node(new_node)
                    distance_to_goal = np.linalg.norm(np.array(new_point)[0:2] - np.array(self.goal)[0:2])
                    # self._node.get_logger().info(f"Distance to goal: {distance_to_goal}")
                    if  distance_to_goal < self.goal_tolerance:
                        # self._node.get_logger().info("Goal reached")
                        self.goal, path_to_goal = self.dubins_steer(new_point, self.goal)
                        if path_to_goal:
                            final_node = Tree_Node(parent = nearest_node,state = self.goal)
                            self.forward_tree.add_node(final_node)
                            break
                    rewired = self.rewiring(new_node)
                    rewire_count += 1 if rewired else 0
                    # self._node.get_logger().info(f"Was tree rewired? {rewired}")
        path, cost = self.forward_tree.find_path(final_node)
        self._node.get_logger().info(f"Generated path with {len(path)} waypoints and rewired {rewire_count} times")
        
        #now we send this path to dubins planner to get additional waypoints.
        dubins_input_forward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path 
                if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None]
        dubins_out_forward, original_indices = sample_complete_plan(dubins_input_forward, self.turning_radius, self.step_dubins, self.obstacles)

        dubins_input_backward = [Waypoint(p[0], p[1], p[3]) for waypoint_node in path_back
                if (p := np.array(waypoint_node.get_state(), dtype=np.float64)) is not None] 
        dubins_out_backward, _ = sample_complete_plan(dubins_input_backward, self.turning_radius, self.step_dubins, self.obstacles)
        # dubins_out = dubins_out_forward + dubins_out_backward[::-1]
        self.visualize_tree(np.array(dubins_out_forward),np.array(dubins_out_backward), path_back)
        
        # convert back to pose2D for transfer
        self.original_wp_indices = [int(i) for i in original_indices]
        goal_msg = self.send_waypoints(dubins_out_forward, path)
        return goal_msg 
    
    def dubins_steer(self, start, end):
        """ Move from start towards end by step_size """
        direction = np.array(end) - np.array(start)
        direction = direction[:2]  # Ignore Z and yaw
        norm = np.linalg.norm(direction)
        if norm == 0:
            return start, False
        # self._node.get_logger().info(f"start: {start} end : {end}")
        # new_point = end
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

    def is_path_collision_free(self, start, end):
        base_heading = end[3]
        step_heading = 6
        headings =  base_heading + np.linspace(0, 360, step_heading, endpoint = False) - 180

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
        for node in self.forward_tree.get_nodes() :
            valid_node = node != new_node and node != self.forward_tree.get_root()
            distance = np.linalg.norm(np.array(new_node.get_state())[0:2] - np.array(node.get_state())[0:2])
            if valid_node and distance < rewiring_distance:
                path_exist = self.is_path_collision_free(new_node.get_state(), node.get_state())  
                
                if  path_exist: #and (distance < self.stepsize)
                    #compute cost of new path and old path
                    # self._node.get_logger().info(f"rewiring node check")
                    old_path,old_cost = self.forward_tree.find_path(node)
                    node.assign_parent(new_node)
                    new_path,new_cost = self.forward_tree.find_path(node)
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
    
    def find_reverse_path(self, state, nearest_node):
        """ Returns the path from the root to the end node."""
        path = []
        reverse_path = []
        if nearest_node == self.forward_tree.get_root():
            reverse_path.append(Tree_Node(state=state, parent=nearest_node))
            return reverse_path
        path, cost = self.forward_tree.find_path(nearest_node)
        # self._node.get_logger().info(f"Path length till neighbor : {len(path)}")
        nearest_node_state = nearest_node.get_state()
        nearest_node_reverse_state = nearest_node_state[0], nearest_node_state[1], nearest_node_state[2], -nearest_node_state[3]
        if self.dubins_steer(state, nearest_node_reverse_state)[1]:
            path.append(Tree_Node(state=state, parent=nearest_node))
            reverse_path = [Tree_Node(state=(p.get_state()[0], p.get_state()[1], p.get_state()[2], -p.get_state()[3]), parent=path[i+1] if i+1 < len(path) else None) for i, p in enumerate(reversed(path))]
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
            goal_waypoint.pose.orientation.z = np.sin(waypoint[2]/2)
            goal_waypoint.pose.orientation.w = np.cos(waypoint[2]/2)
            
            list_waypoints.append(goal_waypoint)
            # waypoint_array.append((waypoint[0], waypoint[1]))  # Ignore Z for 2D plotting

        #plot the waypoints and obstacles
        # waypoint_array = np.array(waypoint_array)
        # self.visualize_tree(waypoint_array, path)

        return list_waypoints
    
    def visualize_tree(self, waypoint_array_forward, waypoint_array_backward, path):
        """ Plot the waypoints and obstacles """
        plt.figure(figsize=(8, 8))
        
        # Plot waypoints
        if len(waypoint_array_forward) > 0:
            # for i in self.original_wp_indices:
            #     plt.scatter(waypoint_array[i][0], waypoint_array[i][1], color='g', marker='x', label="Original Waypoint" if 'Original Waypoint' not in plt.gca().get_legend_handles_labels()[1] else "")
            x, y = waypoint_array_forward[:, 0], waypoint_array_forward[:, 1]
            plt.plot(x, y, marker='o', linestyle='-', color='b', label="Path")
            # plt.scatter(x, y, color='r', label="Waypoints")  # Highlight waypoints
        # if len(waypoint_array_backward) > 0:
        #     x, y = waypoint_array_backward[:, 0], waypoint_array_backward[:, 1]
        #     plt.plot(x, y, marker='o', linestyle='-', color='r', label  = "Reverse Path")
        #     plt.scatter(x, y, color='r')

        # Plot obstacles with their radii
        for ox, oy, _, r in self.obstacles:  # Ignoring Z
            obstacle_circle = plt.Circle((ox, oy), r, color='gray', alpha=0.5, fill=True)
            plt.gca().add_patch(obstacle_circle)
            plt.scatter(ox, oy, color='k', marker='x', label="Obstacle" if 'Obstacle' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        for node in self.forward_tree.get_nodes():
            if node.get_parent() is not None and node not in path:
                parent = node.get_parent()
                plt.plot([node.get_state()[0], parent.get_state()[0]], [node.get_state()[1], parent.get_state()[1]], color='g', linestyle='-', linewidth=0.5)
        # Start and Goal positions
        start_x, start_y = self.start[:2]  # Ignore Z
        goal_x, goal_y = self.goal[:2]  # Ignore Z
        # Plot start and goal
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

    def manager(self):
        """ Periodically check for obstacles and run RRT """
        if not self.get_orientation():
            self._node.get_logger().info("Waiting for initial orientation")
            return
        else:
            if self.start[3] is None:
                self.start = (0, 0, 0, self.get_orientation()) 
                self._node.get_logger().info(f"Initial orientation: {self.start[3]}")
        #get obstacles from tf 
        if not self.obstacles_populated:
            self.get_obstacles_from_tf()
        
        #once you have an obstacle list, run rrt
        if self.obstacles_populated and self.goal_msg == [] and not self.goal_sent:
            self._node.get_logger().info("Obstacles populated. Running RRT")
            self.goal_msg = self.run_rrt(self.start, self.goal)
            #send goal to action client
            self._node.get_logger().info("Sending Goal")
            
            self._ac.waypoint_queue.extend(self.goal_msg[1:])
            self._ac.send_goal()
            self.goal_sent = True
        # self._node.get_logger().info("sent goalpoint.")
            # #cancel timer
            # self._node.get_logger().info("Ending Timer")
            # self.timer.cancel()

def main():
    rclpy.init(args=sys.argv)
    plan_node = rclpy.create_node("rrt_planner")
    planner = RRTPlanner(plan_node)  # Create the planner node

    executor = MultiThreadedExecutor()
    executor.add_node(plan_node)  # Spin planner
    
    try:
        executor.spin()  # Keep both nodes alive
    except KeyboardInterrupt:
        planner._node.get_logger().info("Shutting down")
    finally:
        planner._node.destroy_node()
        # ac_node.destroy_node()
        rclpy.shutdown()
    
if __name__ == "__main__":
    main()