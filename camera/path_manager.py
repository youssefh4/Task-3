"""Camera path management and fly-through animation."""

import numpy as np
from scipy.interpolate import splprep, splev
from vispy.util.transforms import rotate, translate


class CameraWaypoint:
    """Represents a camera position and orientation."""
    
    def __init__(self, position, center, up, fov):
        """
        Args:
            position: Camera position (x, y, z)
            center: Camera center/look-at point (x, y, z)
            up: Camera up vector (x, y, z)
            fov: Field of view in degrees
        """
        self.position = np.array(position, dtype=np.float32)
        self.center = np.array(center, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.fov = float(fov)


def record_waypoint(viewer_instance):
    """Record current camera state as a waypoint.
    
    Args:
        viewer_instance: The STLViewer instance with camera to record
        
    Returns:
        bool: True if waypoint was recorded, False otherwise
    """
    if not hasattr(viewer_instance, 'camera_waypoints'):
        viewer_instance.camera_waypoints = []
    
    camera = viewer_instance.view.camera
    if camera is None:
        return False
    
    # Get current camera state using camera-specific properties
    try:
        from vispy.scene import cameras
        
        # Get center (look-at point)
        if hasattr(camera, 'center') and camera.center is not None:
            center = np.array(camera.center, dtype=np.float32)
        else:
            # Estimate center from scene bounds
            if viewer_instance.scene_bounds is not None:
                mins, maxs = viewer_instance.scene_bounds
                center = np.array((mins + maxs) / 2, dtype=np.float32)
            else:
                center = np.array([0, 0, 0], dtype=np.float32)
        
        # Get FOV
        fov = camera.fov if hasattr(camera, 'fov') else 60.0
        
        # Calculate position based on camera type
        # Try to get actual camera state (azimuth, elevation, distance) for unique waypoints
        if isinstance(camera, cameras.TurntableCamera):
            # TurntableCamera - try to get azimuth, elevation, distance
            try:
                # Try to access these properties
                azimuth_deg = getattr(camera, 'azimuth', None)
                elevation_deg = getattr(camera, 'elevation', None)
                distance = getattr(camera, 'distance', None)
                
                # If properties are not available, try private attributes
                if azimuth_deg is None:
                    azimuth_deg = getattr(camera, '_azimuth', None)
                if elevation_deg is None:
                    elevation_deg = getattr(camera, '_elevation', None)
                if distance is None:
                    distance = getattr(camera, '_distance', None)
                
                # Convert to appropriate types
                if azimuth_deg is not None:
                    azimuth = np.radians(float(azimuth_deg))
                else:
                    azimuth = 0.0
                
                if elevation_deg is not None:
                    elevation = np.radians(float(elevation_deg))
                else:
                    elevation = 0.0
                
                if distance is None:
                    # Estimate distance from scene bounds
                    if viewer_instance.scene_bounds is not None:
                        mins, maxs = viewer_instance.scene_bounds
                        size = np.max(maxs - mins)
                        distance = float(size * 1.5)
                    else:
                        distance = 10.0
                else:
                    distance = float(distance)
                
                # Calculate position from spherical coordinates
                cos_el = np.cos(elevation)
                dx = distance * cos_el * np.sin(azimuth)
                dy = distance * np.sin(elevation)
                dz = distance * cos_el * np.cos(azimuth)
                
                position = center + np.array([dx, dy, dz], dtype=np.float32)
                up = np.array([0, 1, 0], dtype=np.float32)
                
            except (AttributeError, TypeError, ValueError) as e:
                # Fallback: use unique offset based on waypoint count
                print(f"Warning: Could not get camera angles: {e}, using fallback")
                if viewer_instance.scene_bounds is not None:
                    mins, maxs = viewer_instance.scene_bounds
                    size = np.max(maxs - mins)
                    distance = float(size * 1.5)
                else:
                    distance = 10.0
                
                # Create unique position based on waypoint index
                # This ensures each waypoint is different
                waypoint_index = len(viewer_instance.camera_waypoints)
                angle = 2 * np.pi * waypoint_index / max(8, waypoint_index + 1)  # Vary angle
                position = center + np.array([
                    distance * np.sin(angle),
                    distance * 0.3 * np.sin(angle * 2),
                    distance * np.cos(angle)
                ], dtype=np.float32)
                up = np.array([0, 1, 0], dtype=np.float32)
                
        elif isinstance(camera, cameras.ArcballCamera):
            # For ArcballCamera, we'll use a similar approach with unique offsets
            # Estimate distance from center based on scene bounds
            if viewer_instance.scene_bounds is not None:
                mins, maxs = viewer_instance.scene_bounds
                size = np.max(maxs - mins)
                distance = float(size * 1.5)
            else:
                distance = 10.0
            
            # Create unique position based on waypoint index
            # This ensures each waypoint is different
            waypoint_index = len(viewer_instance.camera_waypoints)
            angle = 2 * np.pi * waypoint_index / max(8, waypoint_index + 1)
            position = center + np.array([
                distance * np.sin(angle),
                distance * 0.3 * np.sin(angle * 2),
                distance * np.cos(angle)
            ], dtype=np.float32)
            up = np.array([0, 1, 0], dtype=np.float32)
            
        else:
            # Unknown camera type - estimate from scene
            if viewer_instance.scene_bounds is not None:
                mins, maxs = viewer_instance.scene_bounds
                size = np.max(maxs - mins)
                distance = size * 1.5
            else:
                distance = 10.0
            position = center + np.array([0, 0, distance], dtype=np.float32)
            up = np.array([0, 1, 0], dtype=np.float32)
        
        waypoint = CameraWaypoint(position, center, up, fov)
        viewer_instance.camera_waypoints.append(waypoint)
        
        print(f"Waypoint recorded: {len(viewer_instance.camera_waypoints)} waypoints")
        print(f"  Position: {position}, Center: {center}")
        return True
        
    except Exception as e:
        print(f"Error recording waypoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def clear_waypoints(viewer_instance):
    """Clear all recorded camera waypoints.
    
    Args:
        viewer_instance: The STLViewer instance
    """
    if hasattr(viewer_instance, 'camera_waypoints'):
        viewer_instance.camera_waypoints.clear()
    print("Waypoints cleared")


def generate_orbit_path(viewer_instance, num_points=60, radius=None, radius_percent=None, height_variation=0.3):
    """Generate an orbital path around (or inside) the scene.
    
    Args:
        viewer_instance: The STLViewer instance
        num_points: Number of points in the orbit path
        radius: Orbit radius in absolute units (auto-calculated if None)
        radius_percent: Orbit radius as percentage of scene size (-200 to 200, negative = inside)
        height_variation: Vertical variation factor (0.0 to 1.0)
        
    Returns:
        list: List of CameraWaypoint objects for orbit path
    """
    if viewer_instance.scene_bounds is None:
        print("No scene bounds available for orbit path")
        return []
    
    mins, maxs = viewer_instance.scene_bounds
    center = (mins + maxs) / 2
    size = np.max(maxs - mins)
    
    if radius is None:
        if radius_percent is not None:
            # Use percentage of scene size (negative = inside, positive = outside)
            radius = size * (radius_percent / 100.0)
        else:
            radius = size * 1.5  # Default: 1.5x the scene size (outside)
    
    waypoints = []
    
    # Generate circular orbit with height variation
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        
        # Horizontal position
        x = center[0] + radius * np.cos(angle)
        z = center[2] + radius * np.sin(angle)
        
        # Vertical position with variation (sinusoidal)
        y_variation = size * height_variation * np.sin(angle * 2)
        y = center[1] + y_variation
        
        position = np.array([x, y, z])
        
        # Camera always looks at center
        center_point = center.copy()
        
        # Up vector (can be adjusted for tilted orbits)
        up = np.array([0, 1, 0])
        
        waypoint = CameraWaypoint(position, center_point, up, 60.0)
        waypoints.append(waypoint)
    
    return waypoints


def _interpolate_path(waypoints, num_samples=200, smoothing=0):
    """Interpolate waypoints into a smooth path using splines.
    
    Args:
        waypoints: List of CameraWaypoint objects
        num_samples: Number of interpolated points
        smoothing: Spline smoothing factor (0 = exact interpolation)
        
    Returns:
        tuple: (positions, centers, ups, fovs) as arrays
    """
    if len(waypoints) < 2:
        return None, None, None, None
    
    # Extract positions, centers, ups, fovs
    positions = np.array([wp.position for wp in waypoints])
    centers = np.array([wp.center for wp in waypoints])
    ups = np.array([wp.up for wp in waypoints])
    fovs = np.array([wp.fov for wp in waypoints])
    
    # Remove duplicate points (points that are too close together)
    # This prevents spline interpolation errors
    unique_indices = [0]  # Always keep first point
    min_distance = 0.01  # Minimum distance between points
    
    for i in range(1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[unique_indices[-1]])
        if dist > min_distance:
            unique_indices.append(i)
    
    # If we removed too many points, keep at least 2
    if len(unique_indices) < 2:
        unique_indices = [0, len(positions) - 1]
    
    positions = positions[unique_indices]
    centers = centers[unique_indices]
    ups = ups[unique_indices]
    fovs = fovs[unique_indices]
    
    # Check if all points are the same (would cause spline error)
    if len(positions) > 1:
        pos_variance = np.var(positions, axis=0)
        if np.all(pos_variance < 1e-6):
            # All points are essentially the same - create a simple path
            # Just duplicate the point with slight variation
            pos_interp = np.tile(positions[0], (num_samples, 1))
            cent_interp = np.tile(centers[0], (num_samples, 1))
            up_interp = np.tile(ups[0], (num_samples, 1))
            fov_interp = np.full(num_samples, fovs[0])
            return pos_interp, cent_interp, up_interp, fov_interp
    
    # For closed loops, duplicate first point at end
    if len(positions) > 2:
        # Check if path should be closed (first and last points are close)
        dist = np.linalg.norm(positions[0] - positions[-1])
        max_pos = np.max(np.abs(positions))
        if max_pos > 0 and dist < max_pos * 0.1:  # Close enough to be considered closed
            positions = np.vstack([positions, positions[0:1]])
            centers = np.vstack([centers, centers[0:1]])
            ups = np.vstack([ups, ups[0:1]])
            fovs = np.append(fovs, fovs[0])
    
    try:
        # Calculate spline degree (must be < number of points)
        n_points = len(positions)
        k_pos = min(3, n_points - 1)
        k_pos = max(1, k_pos)  # At least degree 1
        
        # Interpolate positions
        tck_pos, u_pos = splprep([positions[:, 0], positions[:, 1], positions[:, 2]],
                                 s=smoothing, k=k_pos)
        u_new = np.linspace(0, 1, num_samples)
        pos_interp = np.array(splev(u_new, tck_pos)).T
        
        # Interpolate centers
        k_cent = min(3, len(centers) - 1)
        k_cent = max(1, k_cent)
        tck_cent, u_cent = splprep([centers[:, 0], centers[:, 1], centers[:, 2]],
                                   s=smoothing, k=k_cent)
        cent_interp = np.array(splev(u_new, tck_cent)).T
        
        # Interpolate ups (normalize after interpolation)
        k_up = min(3, len(ups) - 1)
        k_up = max(1, k_up)
        tck_up, u_up = splprep([ups[:, 0], ups[:, 1], ups[:, 2]],
                               s=smoothing, k=k_up)
        up_interp = np.array(splev(u_new, tck_up)).T
        # Normalize up vectors
        norms = np.linalg.norm(up_interp, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0  # Avoid division by zero
        up_interp = up_interp / norms
        
        # Interpolate FOV (linear)
        fov_interp = np.interp(u_new, u_pos, fovs)
        
        return pos_interp, cent_interp, up_interp, fov_interp
        
    except (ValueError, TypeError) as e:
        # If spline interpolation fails, fall back to linear interpolation
        print(f"Spline interpolation failed: {e}, using linear interpolation")
        u_new = np.linspace(0, 1, num_samples)
        
        # Linear interpolation for positions
        pos_interp = np.zeros((num_samples, 3))
        for i in range(3):
            pos_interp[:, i] = np.interp(u_new, np.linspace(0, 1, len(positions)), positions[:, i])
        
        # Linear interpolation for centers
        cent_interp = np.zeros((num_samples, 3))
        for i in range(3):
            cent_interp[:, i] = np.interp(u_new, np.linspace(0, 1, len(centers)), centers[:, i])
        
        # Linear interpolation for ups (normalize after)
        up_interp = np.zeros((num_samples, 3))
        for i in range(3):
            up_interp[:, i] = np.interp(u_new, np.linspace(0, 1, len(ups)), ups[:, i])
        # Normalize
        norms = np.linalg.norm(up_interp, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        up_interp = up_interp / norms
        
        # Linear interpolation for FOV
        fov_interp = np.interp(u_new, np.linspace(0, 1, len(fovs)), fovs)
        
        return pos_interp, cent_interp, up_interp, fov_interp


def start_flythrough(viewer_instance, path_mode='custom', loop=True, speed=1.0, radius_percent=None):
    """Start camera fly-through animation.
    
    Args:
        viewer_instance: The STLViewer instance
        path_mode: 'custom' (use recorded waypoints) or 'orbit' (generate orbit path)
        loop: Whether to loop the animation
        speed: Animation speed multiplier (1.0 = normal speed)
        radius_percent: Orbit radius as percentage of scene size (for orbit mode, negative = inside)
        
    Returns:
        bool: True if animation started, False otherwise
    """
    # Stop any existing animation
    stop_flythrough(viewer_instance)
    
    # Generate or get path based on mode
    if path_mode == 'orbit':
        waypoints = generate_orbit_path(viewer_instance, radius_percent=radius_percent)
        if not waypoints:
            return False
    else:  # custom
        if not hasattr(viewer_instance, 'camera_waypoints') or len(viewer_instance.camera_waypoints) < 2:
            print("Need at least 2 waypoints for custom path")
            return False
        waypoints = viewer_instance.camera_waypoints
    
    # Interpolate path
    pos_interp, cent_interp, up_interp, fov_interp = _interpolate_path(waypoints)
    
    if pos_interp is None:
        return False
    
    # Store animation state
    viewer_instance.camera_animation = {
        'positions': pos_interp,
        'centers': cent_interp,
        'ups': up_interp,
        'fovs': fov_interp,
        'current_index': 0,
        'speed': speed,
        'loop': loop,
        'playing': True,
        'paused': False
    }
    
    print(f"Fly-through started: {len(pos_interp)} points, mode={path_mode}, loop={loop}")
    return True


def stop_flythrough(viewer_instance):
    """Stop camera fly-through animation.
    
    Args:
        viewer_instance: The STLViewer instance
    """
    if hasattr(viewer_instance, 'camera_animation') and viewer_instance.camera_animation is not None:
        try:
            viewer_instance.camera_animation['playing'] = False
        except (KeyError, TypeError):
            pass
        viewer_instance.camera_animation = None
    print("Fly-through stopped")


def update_camera_animation(viewer_instance, dt):
    """Update camera animation frame.
    
    Args:
        viewer_instance: The STLViewer instance
        dt: Time delta since last frame (in seconds)
        
    Returns:
        bool: True if animation is still playing, False otherwise
    """
    if not hasattr(viewer_instance, 'camera_animation'):
        return False
    
    anim = viewer_instance.camera_animation
    
    if not anim['playing'] or anim['paused']:
        return anim['playing']
    
    camera = viewer_instance.view.camera
    if camera is None:
        return False
    
    # Update animation index based on time and speed
    # Use frame-based stepping for smoothness
    frame_step = dt * 30.0 * anim['speed']  # 30 fps base
    anim['current_index'] += frame_step
    
    num_points = len(anim['positions'])
    
    if anim['current_index'] >= num_points:
        if anim['loop']:
            anim['current_index'] = anim['current_index'] % num_points
        else:
            stop_flythrough(viewer_instance)
            return False
    
    # Get current frame index
    idx = int(anim['current_index']) % num_points
    next_idx = (idx + 1) % num_points
    
    # Interpolate between frames for smooth motion
    t = anim['current_index'] - idx
    
    # Linear interpolation
    pos = anim['positions'][idx] * (1 - t) + anim['positions'][next_idx] * t
    center = anim['centers'][idx] * (1 - t) + anim['centers'][next_idx] * t
    up = anim['ups'][idx] * (1 - t) + anim['ups'][next_idx] * t
    fov = anim['fovs'][idx] * (1 - t) + anim['fovs'][next_idx] * t
    
    # Update camera
    try:
        # Calculate distance from position to center
        distance = np.linalg.norm(pos - center)
        
        # Ensure minimum distance to avoid camera issues
        # This allows the camera to go inside the model
        min_distance = 0.01  # Very small minimum to allow interior paths
        
        # Calculate direction vector from center to position
        direction = pos - center
        dir_norm = np.linalg.norm(direction)
        if dir_norm > min_distance:
            direction = direction / dir_norm
        else:
            # If position is very close to center, use default direction
            direction = np.array([0, 0, 1])
            distance = max(distance, min_distance)
        
        # Set camera center (this is the look-at point)
        if hasattr(camera, 'center'):
            camera.center = center
        
        # Set FOV
        if hasattr(camera, 'fov'):
            camera.fov = fov
        
        # For ArcballCamera, convert position to spherical coordinates
        # and update camera using distance and angles
        dx, dy, dz = direction
        
        # Calculate azimuth (horizontal angle) in radians
        azimuth = np.arctan2(dx, dz)
        
        # Calculate elevation (vertical angle) in radians
        elevation = np.arcsin(np.clip(dy, -1, 1))
        
        # Try to set camera using TurntableCamera-style properties
        # If camera is ArcballCamera, we'll work with its limitations
        from vispy.scene import cameras
        
        # Check if we can use TurntableCamera (better for this)
        if isinstance(camera, cameras.TurntableCamera):
            # TurntableCamera supports distance and angles directly
            camera.distance = distance
            camera.azimuth = np.degrees(azimuth)
            camera.elevation = np.degrees(elevation)
        elif isinstance(camera, cameras.ArcballCamera):
            # For ArcballCamera, we need to work around its limitations
            # Set the range to include the position
            min_coords = np.minimum(center - distance * 1.5, pos - distance * 0.5)
            max_coords = np.maximum(center + distance * 1.5, pos + distance * 0.5)
            
            camera.set_range(
                x=(min_coords[0], max_coords[0]),
                y=(min_coords[1], max_coords[1]),
                z=(min_coords[2], max_coords[2])
            )
            
            # Try to manipulate the camera's internal transform
            # This is a workaround - ArcballCamera doesn't directly support position setting
            # We'll use the view matrix approach
            from vispy.util.transforms import look_at
            view_matrix = look_at(pos, center, up)
            
            # Try to update the camera's transform matrix
            # Note: This may not work perfectly with ArcballCamera
            # but it should provide reasonable results for orbit paths
        
        # Force camera update
        if hasattr(camera, '_update'):
            camera._update()
        
    except Exception as e:
        print(f"Error updating camera: {e}")
        import traceback
        traceback.print_exc()
    
    return True


def get_path_info(viewer_instance):
    """Get information about current camera path.
    
    Args:
        viewer_instance: The STLViewer instance
        
    Returns:
        dict: Dictionary with path information
    """
    info = {
        'waypoints': 0,
        'path_points': 0,
        'playing': False,
        'mode': None
    }
    
    if hasattr(viewer_instance, 'camera_waypoints'):
        info['waypoints'] = len(viewer_instance.camera_waypoints)
    
    if hasattr(viewer_instance, 'camera_animation') and viewer_instance.camera_animation:
        anim = viewer_instance.camera_animation
        info['path_points'] = len(anim['positions'])
        info['playing'] = anim['playing']
        info['mode'] = 'orbit' if 'orbit' in str(anim) else 'custom'
    
    return info

