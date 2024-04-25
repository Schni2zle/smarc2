#!/usr/bin/python3

from .waypoint import UnderwaterWaypoint
from smarc_mission_msgs.msg import GotoWaypoint

class SMaRCWP(UnderwaterWaypoint):
    def __init__(self, goto_waypoint: GotoWaypoint) -> None:
        super().__init__()

        self._goto_wp = goto_waypoint

    @property
    def name(self) -> str:
        return self._goto_wp.name

    @property
    def is_actionable(self) -> bool:
        """
        Check that the WP is not all 0s...
        """
        x,y,z = self.position
        d = self.depth
        a = self.altitude
        if all([x==0, y==0, z==0, d==0, a==0]): return False
        return True
    
    @property
    def position(self) -> tuple[float, float, float]:
        p = self._goto_wp.pose.pose.position
        return (p.x, p.y, p.z)
    
    @property
    def reference_frame(self) -> str:
        return self._goto_wp.pose.header.frame_id
    
    @property
    def depth(self) -> float:
        return self._goto_wp.travel_depth
    
    @property
    def altitude(self) -> float:
        return self._goto_wp.travel_altitude
    
    