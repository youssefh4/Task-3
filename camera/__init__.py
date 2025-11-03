"""Camera path and fly-through animation module."""

from camera.path_manager import (
    record_waypoint, clear_waypoints, generate_orbit_path,
    start_flythrough, stop_flythrough, update_camera_animation,
    get_path_info
)

__all__ = [
    'record_waypoint', 'clear_waypoints', 'generate_orbit_path',
    'start_flythrough', 'stop_flythrough', 'update_camera_animation',
    'get_path_info'
]

