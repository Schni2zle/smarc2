from smarc_bt.vehicles.sam_auv import SAMAuv
from rclpy.node import Node
from smarc_bt.vehicles.sensor import SensorNames
class StateInformation() :
    """maintains all sensor information of SAM"""
    def __init__(self,
                 node: Node) -> None:
        self._sam = SAMAuv(node)
        self._node = node
        self._node.get_logger().info("StateInformation initialized")

        #in8itializing relevant states
        self.battery = 0
        self.depth = 0
        self.altitude = 0
        self.global_position = ""
        self.healthy = False

        # Timer to check state periodically
        # self._node.create_timer(1.0, self.check_state)
        

    def check_state(self):
        """Fetch sensor states from SAMAuv."""
        vehicle_state = self._sam.vehicle_state

        self.depth = vehicle_state[SensorNames.DEPTH]._values
        self.altitude = vehicle_state[SensorNames.ALTITUDE]._values
        self.global_position = vehicle_state[SensorNames.GLOBAL_POSITION]._values
        self.healthy = vehicle_state[SensorNames.VEHICLE_HEALTHY]._values
        self.battery = vehicle_state[SensorNames.BATTERY]._values

        # self._node.get_logger().info(f"Depth: {self.depth}m, Altitude: {self.altitude}m, Position: {self.global_position}, Healthy: {self.healthy}, Battery: {self.battery}")
    