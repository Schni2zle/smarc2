#!/usr/bin/python3

import rclpy, sys, random
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy import time

from action_msgs.msg import GoalStatus
from smarc_mission_msgs.action import GotoWaypoint
from smarc_mission_msgs.msg import Topics as MissionTopics

from geometry_msgs.msg import PoseStamped

# ROS imports
from builtin_interfaces.msg import Time as Stamp
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time as rcl_Time
from rclpy.executors import MultiThreadedExecutor

class DiveToWaypointActionClient():
    """
    Action Client version of the setpoint node

    A client to call the dive controller server and use it to follow planners waypoints.
    This is basically an annotated version of:
    https://github.com/ros2/examples/tree/humble/rclpy/actions/minimal_action_client
    You can find some other examples there that do slightly different things.
    """
    def __init__(self,
                 node: Node) -> None:
        self._node = node

        self._ac = ActionClient(node=self._node,
                                action_type=GotoWaypoint,
                                action_name=f'/sam_auv_v1/{MissionTopics.GOTO_WP_ACTION}')
        self._loginfo(f"Action Client created: {f'/sam_auv_v1/{MissionTopics.GOTO_WP_ACTION}'}")
        # to check if we even have a running server later
        self._goal_handle = None
        self.current_waypoint = None
        self.next_waypoint = None
        self.waypoint_queue = []

        #subscribing to next waypoint
        # self._node.create_subscription(PoseStamped, f'/sam_auv_v1/{MissionTopics.WAYPOINT_TOPIC}', self.waypoint_cb, 10)

    def _loginfo(self, s):
        self._node.get_logger().info(s)

    # def waypoint_cb(self, msg):
    #     """Handles incoming waypoints (if using dynamic waypoints from a topic)."""
    #     self._loginfo(f"Received waypoint: {msg}")
    #     self.waypoint_queue.append(msg)
    #     if self.current_waypoint is None:
    #         self.assign_next_waypoint()
    #         self.send_goal()


    def _goal_response_cb(self, future):
        """
        We will register this method to the action client when we
        send a goal later.
        The action client will call this when we get a response from
        the server: It could accept or reject our goal, the future here
        will have that information for us to handle.
        """
        goal_handle = future.result() # wait for it
        if not goal_handle.accepted:
            self._loginfo("Goal rejected. Sadness")
            return

        self._loginfo("Goal accepted. Success, hurray")

        self._goal_handle = goal_handle

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_cb)


    def _get_result_cb(self, future):
        result = future.result().result
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._loginfo(f"Success: {result.reached_waypoint}. Assigning next waypoint")
            if self.assign_next_waypoint():
                self.send_goal()
        else:
            self._loginfo(f"NOT Success: Status:{status}, result:{result.reached_waypoint}")

        # rclpy.shutdown()

    def assign_next_waypoint(self):
        if self.waypoint_queue :
            self.current_waypoint = self.waypoint_queue.pop(0)
            return True
        else:
            self._loginfo("No more waypoints to send. Goal reached????")
            return False



    def _feedback_cb(self, feedback):
        # this field contains the object we defined in smarc_mission_msgs/action/GotoWaypoint.action
        self._loginfo(f"Got feedback from server: {feedback.feedback.feedback_message}")

    def send_goal(self):

        self._loginfo("Waiting for server to come alive")
        server_is_ready = self._ac.wait_for_server(timeout_sec=30)

        if not server_is_ready:
            self._loginfo("Server was not availble, quitting!")
            rclpy.shutdown()
            return

        goal_msg = GotoWaypoint.Goal()
        if self.current_waypoint is None:
            self.current_waypoint = PoseStamped()
            self.current_waypoint = self.waypoint_queue.pop(0)
            #if pop return None should I make a special bool to say all the waypoints been covered?
        goal_msg.waypoint.pose = self.current_waypoint
        goal_msg.waypoint.pose.header.frame_id = 'sam_auv_v1/odom_gt'
        
        goal_msg.waypoint.pose.header.stamp = self.rcl_time_to_stamp(self._node.get_clock().now())
        #TODO: decide if these values should be assigned here
        goal_msg.waypoint.travel_rpm = 500.0
        goal_msg.waypoint.goal_tolerance = 1.0
        
        self._loginfo(f"Sending goal: {goal_msg}")

        self._send_goal_future = self._ac.send_goal_async(
            goal=goal_msg,
            feedback_callback=self._feedback_cb)

        self._send_goal_future.add_done_callback(self._goal_response_cb)


    def rcl_time_to_stamp(self,time: rcl_Time) -> Stamp:
        """
        Converts rcl Time to stamp
        :param time:
        :return:
        """
        stamp = Stamp()
        stamp.sec = int(time.nanoseconds // 1e9)
        stamp.nanosec = int(time.nanoseconds % 1e9)
        return stamp

    def cancel_goal(self):
        self._loginfo("Cancel Goal")

        self._loginfo(f"Goal Handle: {self._goal_handle}")

        if self._goal_handle is not None:
            self._loginfo('Sending cancel request...')
            cancel_future = self._goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_response_callback)

    def cancel_response_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self._loginfo('Goal successfully cancelled')
        else:
            self._loginfo('Goal failed to cancel')


def main():
    # create a node and our objects in the usual manner.
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("DiveActionClientNode")

    ac = DiveToWaypointActionClient(node)

    # Add test waypoints manually
    test_waypoints = [
        PoseStamped(),
        PoseStamped(),
        PoseStamped()
    ]
    test_waypoints[0].pose.position.x = 10.0
    test_waypoints[0].pose.position.y = 0.0
    test_waypoints[0].pose.position.z = -3.0

    test_waypoints[1].pose.position.x = 20.0
    test_waypoints[1].pose.position.y = 0.0
    test_waypoints[1].pose.position.z = -5.0

    test_waypoints[2].pose.position.x = 30.0
    test_waypoints[2].pose.position.y = 0.0
    test_waypoints[2].pose.position.z = -7.0

    ac.waypoint_queue.extend(test_waypoints)

    ac.send_goal()

    # # To test the cancel callback
    # node.create_timer(5.0, ac.cancel_goal)
    # executor = MultiThreadedExecutor()
    try: 
        rclpy.spin(node )
        ac._loginfo("spin")
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
