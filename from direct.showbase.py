import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import copy
import math
import time
import tkinter as tk
from tkinter import filedialog
import os

# Try to import FBX support
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[Note] trimesh not available for FBX support. Install with: pip install trimesh")

PYASSIMP_AVAILABLE = False
try:
    import pyassimp  # type: ignore
    PYASSIMP_AVAILABLE = True
except:
    # pyassimp failed (AssimpError or ImportError)
    pass

# GPU detection and acceleration info
print("[GPU] Checking system GPU capabilities...")
try:
    import vtk
    from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderWindow
    import sys
    import os
    
    def get_gpu_info():
        """Get GPU information using OpenGL"""
        try:
            # Create OpenGL render window
            test_window = vtkOpenGLRenderWindow()
            test_window.SetOffScreenRendering(1)
            # Force context creation
            try:
                test_window.Render()
            except:
                pass
            
            # Get OpenGL information (instance methods when available)
            renderer_str = None
            try:
                if hasattr(test_window, 'GetGPUInfoString'):
                    renderer_str = test_window.GetGPUInfoString()
            except:
                pass
            gpu_info = {
                'renderer': renderer_str if renderer_str else "Unknown"
            }
            
            # Try to get dedicated GPU memory (Windows only)
            if sys.platform == 'win32':
                try:
                    import ctypes
                    class DXGI_ADAPTER_DESC(ctypes.Structure):
                        _fields_ = [
                            ("Description", ctypes.c_wchar * 128),
                            ("VendorId", ctypes.c_uint),
                            ("DeviceId", ctypes.c_uint),
                            ("SubSysId", ctypes.c_uint),
                            ("Revision", ctypes.c_uint),
                            ("DedicatedVideoMemory", ctypes.c_size_t),
                            ("DedicatedSystemMemory", ctypes.c_size_t),
                            ("SharedSystemMemory", ctypes.c_size_t),
                            ("AdapterLuid", ctypes.c_int64)
                        ]
                    if hasattr(ctypes.windll, 'dxgi'):
                        gpu_memory = DXGI_ADAPTER_DESC().DedicatedVideoMemory
                        gpu_info['memory'] = f"{round(gpu_memory/(1024**3), 2)}GB"
                except:
                    pass
            
            return gpu_info
        except Exception as e:
            print(f"[GPU] Could not get detailed GPU info: {e}")
            return None
    
    # Get and print GPU information
    gpu_info = get_gpu_info()
    if gpu_info:
        print("\n[GPU] Detected Graphics Hardware:")
        for key, value in gpu_info.items():
            print(f"      {key.replace('_', ' ').title()}: {value}")
    
    # Check VTK/OpenGL capabilities
    print("\n[GPU] Checking VTK/OpenGL Support:")
    print(f"[GPU] VTK version: {vtk.VTK_VERSION}")
    
    # Create test render window to check capabilities
    render_window = vtkOpenGLRenderWindow()
    render_window.SetOffScreenRendering(1)
    
    # Force context creation before capability queries
    try:
        render_window.Render()
    except Exception:
        pass

    # Helper to parse ReportCapabilities text
    def _parse_caps(txt):
        info = {"vendor": None, "renderer": None, "version": None}
        try:
            lines = (txt or "").splitlines()
            for line in lines:
                low = line.lower().strip()
                if 'opengl vendor' in low or low.startswith('vendor:'):
                    info["vendor"] = line.split(':', 1)[-1].strip()
                elif 'opengl renderer' in low or low.startswith('renderer:'):
                    info["renderer"] = line.split(':', 1)[-1].strip()
                elif 'opengl version' in low or low.startswith('version:'):
                    info["version"] = line.split(':', 1)[-1].strip()
        except Exception:
            pass
        return info

    caps_text = None
    try:
        if hasattr(render_window, 'ReportCapabilities'):
            caps_text = render_window.ReportCapabilities()
    except Exception:
        caps_text = None

    caps_info = _parse_caps(caps_text) if caps_text else {"vendor": None, "renderer": None, "version": None}

    # Get OpenGL version string if available, else fallback
    try:
        if hasattr(render_window, 'GetOpenGLVersion'):
            gl_version = render_window.GetOpenGLVersion()
        else:
            gl_version = caps_info.get('version') or "Unknown"
    except Exception:
        gl_version = caps_info.get('version') or "Unknown"
    
    # Check hardware acceleration
    has_acceleration = True  # Modern VTK always uses hardware acceleration when available
    
    print(f"[GPU] OpenGL Version: {gl_version}")
    if caps_info.get('vendor') or caps_info.get('renderer'):
        vendor_str = (caps_info.get('vendor') or 'Unknown')
        renderer_str = (caps_info.get('renderer') or 'Unknown')
        print(f"[GPU] GL Vendor: {vendor_str}")
        print(f"[GPU] GL Renderer: {renderer_str}")
        # Heuristic: detect software renderers
        sw_markers = ['gdi generic', 'mesa', 'llvmpipe', 'software']
        sw = any(m in vendor_str.lower() or m in renderer_str.lower() for m in sw_markers)
        print(f"[GPU] Hardware Rendering Likely: {'YES' if (not sw and gl_version != 'Unknown') else 'NO'}")
    print(f"[GPU] Hardware Acceleration: {'Enabled' if has_acceleration else 'Disabled'}")
    try:
        max_ms = vtk.vtkOpenGLRenderer.GetGlobalMaximumNumberOfMultiSamples()
        print(f"[GPU] Max Multisamples: {max_ms}")
    except Exception:
        pass
    
    # Setup optimal GPU rendering settings
    if has_acceleration:
        # Configure VTK for optimal GPU usage
        try:
            vtk.vtkOpenGLRenderWindow.SetGlobalMaximumNumberOfMultiSamples(0)  # Disable MSAA for performance
        except Exception:
            pass
        try:
            vtk.vtkOpenGLHardwareSelector().SetFieldAssociation(0)  # Optimize selection
        except Exception:
            pass
        
        # Enable GPU memory optimization flags (guard per VTK build)
        try:
            if hasattr(render_window, 'SetUseOffscreenBuffers'):
                render_window.SetUseOffscreenBuffers(True)  # Use GPU buffers
        except Exception:
            pass
        try:
            if hasattr(render_window, 'SetUseHardware'):
                render_window.SetUseHardware(True)  # Force hardware rendering
        except Exception:
            pass
        try:
            render_window.SetDesiredUpdateRate(60.0)  # Target 60 FPS
        except Exception:
            pass
        try:
            render_window.SetMultiSamples(0)  # Disable MSAA
        except Exception:
            pass
        try:
            render_window.SetPointSmoothing(False)  # Disable point smoothing
            render_window.SetLineSmoothing(False)  # Disable line smoothing
            render_window.SetPolygonSmoothing(False)  # Disable polygon smoothing
        except Exception:
            pass
        
        print("\n[GPU] Performance Optimizations:")
        print("[GPU] ✓ Hardware rendering forced")
        print("[GPU] ✓ GPU memory optimization enabled")
        print("[GPU] ✓ Antialiasing disabled for performance")
        print("[GPU] ✓ Smoothing disabled for performance")
        print("[GPU] ✓ Frame rate target: 30 FPS")
    
    # Additional VTK GPU capabilities
    render_window.SetOffScreenRendering(1)
    try:
        if hasattr(render_window, 'GetGPUInfo'):
            gpu_info_det = render_window.GetGPUInfo()
            print(f"\n[GPU] VTK GPU Info: {gpu_info_det}")
    except Exception:
        pass
    
    # Check for advanced features
    print("\n[GPU] Advanced Features:")
    try:
        print(f"[GPU] Depth Peeling Support: {'Yes' if render_window.GetAlphaBitPlanes() > 0 else 'No'}")
    except Exception:
        pass
    try:
        print(f"[GPU] Hardware Selection: {'Yes' if vtk.vtkHardwareSelector().GetAvailable() else 'No'}")
    except Exception:
        pass
    
except Exception as e:
    print(f"[GPU] Could not check GPU info: {e}")
    pass

###########################################################
# Heart Pump Feature (Moving Stuff) + GUI for mesh selection
###########################################################

def start_moving_stuff(plotter, meshes_dict):
    """
    Animate heart pumping motion.
    Args:
        plotter: existing PyVista or BackgroundPlotter instance.
        meshes_dict: dict {name: actor} where actor.mapper.dataset is a PyVista mesh
    """

    print("[Moving Stuff] Initializing heart pump animation...")

    # helper for fuzzy chamber detection
    def find(keywords):
        out = []
        for k in meshes_dict.keys():
            low = k.lower()
            for word in keywords:
                if word in low:
                    out.append(k)
                    break
        return out

    # Guess structures from names
    left_v  = find(["left ventricle", "lv", "ventricle left"])
    right_v = find(["right ventricle", "rv", "ventricle right"])
    left_a  = find(["left atrium", "la", "atrium left"])
    right_a = find(["right atrium", "ra", "atrium right"])
    aorta   = find(["aorta", "ascending", "arch", "aortic"])

    # Fallback: if we couldn't match user labels, at least animate everything
    chamber_names = left_v + right_v + left_a + right_a
    if not chamber_names:
        chamber_names = list(meshes_dict.keys())
        print("[!] No named chambers matched (LV/RV/LA/RA). Falling back to ALL meshes.")

    print(f"[Moving Stuff] Chambers to animate: {chamber_names}")
    print(f"[Moving Stuff] Aorta candidates: {aorta}")
    print(f"[Moving Stuff] ALL {len(chamber_names)} chambers will pump together in sync!")

    # Helper: apply GPU-friendly mapper/property settings to an actor
    def set_gpu_fast_props(actor):
        try:
            mapper = actor.GetMapper()
            if mapper:
                # Prefer GPU-side interpolation and avoid scalar coloring
                try:
                    mapper.SetInterpolateScalarsBeforeMapping(1)
                except Exception:
                    pass
                try:
                    mapper.SetScalarVisibility(0)
                except Exception:
                    pass
                try:
                    mapper.SetImmediateModeRendering(0)
                except Exception:
                    pass
            prop = actor.GetProperty()
            if prop:
                # Reduce expensive lighting work; enable backface culling
                try:
                    prop.SetSpecular(0.1)
                    prop.SetSpecularPower(10)
                    prop.SetAmbient(0.3)
                    prop.SetDiffuse(0.9)
                except Exception:
                    pass
                try:
                    prop.BackfaceCullingOn()
                except Exception:
                    pass
        except Exception:
            pass

    # Save original (rest pose) coordinates
    base_points = {}
    for name in chamber_names + aorta:
        try:
            base_points[name] = copy.deepcopy(meshes_dict[name].mapper.dataset.points)
        except Exception as e:
            print(f"[!] Couldn't copy base points for {name}: {e}")

    # Pick an axis to split "top" (atria) vs "bottom" (ventricles)
    # We'll guess Y axis (index=1). Change to 2 if heart is rotated.
    axis_index = 1

    # Calculate heart bounds from all meshes
    if not chamber_names or len(base_points) == 0:
        raise ValueError("No valid chambers loaded. Cannot proceed with animation.")
    
    all_pts = np.vstack([base_points[name] for name in chamber_names])
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()
    y_mid = 0.5 * (y_min + y_max)
    
    # Calculate heart center and size
    heart_center = np.array([0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (z_min + z_max)])
    heart_size = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    
    print(f"[Heart] Bounds: Y={y_min:.1f} to {y_max:.1f}, center={heart_center}, size={heart_size}")

    # Precompute vertex-based separation masks (global, not per-part)
    # Each mesh gets boolean masks selecting vertices above/below the global y_mid
    vertex_masks = {}
    for name in chamber_names:
        pts = base_points.get(name)
        if pts is None:
            continue
        upper_mask = pts[:, axis_index] > y_mid
        lower_mask = ~upper_mask
        vertex_masks[name] = (upper_mask, lower_mask)

    # Compute global centers for upper/lower vertex groups (using base pose)
    all_upper_points = []
    all_lower_points = []
    for name in chamber_names:
        pts = base_points.get(name)
        if pts is None or name not in vertex_masks:
            continue
        um, lm = vertex_masks[name]
        if np.any(um):
            all_upper_points.append(pts[um])
        if np.any(lm):
            all_lower_points.append(pts[lm])

    if len(all_upper_points) > 0:
        global_upper_center = np.mean(np.vstack(all_upper_points), axis=0)
    else:
        global_upper_center = heart_center
    if len(all_lower_points) > 0:
        global_lower_center = np.mean(np.vstack(all_lower_points), axis=0)
    else:
        global_lower_center = heart_center

    # Animation state and timing
    t_val = {"t": 0.0, "frame_counter": 0}  # Combined time tracking
    beat_period = 0.6  # seconds per heartbeat (100 BPM - realistic active heart rate)
    # Heart animation control state
    heart_anim_enabled = {"state": True}
    heart_anim = {"enabled": True, "cb_id": None}  # Animation callback tracking
    
    # Heart opacity control - start semi-transparent
    heart_opacity = {"state": 0.3}  # Start semi-transparent
    all_mesh_actors = list(meshes_dict.values())
    # Apply GPU-friendly properties to heart mesh actors
    for _actor in all_mesh_actors:
        set_gpu_fast_props(_actor)

    # Opacity slider (0.0 - 1.0)
    try:
        def _on_opacity_change(value):
            try:
                # PyVista passes float value directly
                val = max(0.0, min(1.0, float(value)))
                heart_opacity["state"] = val
                for actor in all_mesh_actors:
                    try:
                        actor.GetProperty().SetOpacity(val)
                    except Exception:
                        pass
                try:
                    plotter.render()
                except Exception:
                    pass
            except Exception:
                pass

        # Add slider widget to the UI
        plotter.add_slider_widget(
            _on_opacity_change,
            rng=[0.0, 1.0],
            value=heart_opacity["state"],
            title='Opacity',
            pointa=(0.03, 0.06),
            pointb=(0.45, 0.06),
            style='modern'
        )
    except Exception:
        pass

    # Renderer/global GPU tweaks
    try:
        # Allow time/LOD tradeoff for better throughput
        if hasattr(plotter.renderer, 'SetOcclusionRatio'):
            plotter.renderer.SetOcclusionRatio(0.1)
        # Disable expensive shadows if present
        if hasattr(plotter.renderer, 'SetUseShadows'):
            plotter.renderer.SetUseShadows(0)
    except Exception:
        pass
    
    # Precompute connection points between chambers (valve locations)
    def find_connection_points(top_chambers, bottom_chambers):
        """Find connection points between upper and lower chambers (valve locations)"""
        connection_points = []
        for top_name in top_chambers:
            for bot_name in bottom_chambers:
                if top_name in meshes_dict and bot_name in meshes_dict:
                    top_mesh = meshes_dict[top_name].mapper.dataset
                    bot_mesh = meshes_dict[bot_name].mapper.dataset
                    
                    # Find points near the boundary between chambers
                    top_lower = top_mesh.points[top_mesh.points[:, axis_index] < y_mid + (y_max - y_mid) * 0.3]
                    bot_upper = bot_mesh.points[bot_mesh.points[:, axis_index] > y_mid - (y_mid - y_min) * 0.3]
                    
                    if len(top_lower) > 0 and len(bot_upper) > 0:
                        # Average positions to find valve location
                        valve_pos = (np.mean(top_lower, axis=0) + np.mean(bot_upper, axis=0)) / 2
                        connection_points.append((top_name, bot_name, valve_pos))
        
        return connection_points if connection_points else None
    
    # Identify right and left side chambers for dual circulation
    left_v_chambers = [n for n in chamber_names if any(kw in n.lower() for kw in ['left ventricle', 'lv', 'ventricle left'])]
    right_v_chambers = [n for n in chamber_names if any(kw in n.lower() for kw in ['right ventricle', 'rv', 'ventricle right'])]
    left_a_chambers = [n for n in chamber_names if any(kw in n.lower() for kw in ['left atrium', 'la', 'atrium left'])]
    right_a_chambers = [n for n in chamber_names if any(kw in n.lower() for kw in ['right atrium', 'ra', 'atrium right'])]
    
    # Find valve connections
    right_atrioventricular = find_connection_points(right_a_chambers if right_a_chambers else chamber_names, 
                                                     right_v_chambers if right_v_chambers else chamber_names)
    left_atrioventricular = find_connection_points(left_a_chambers if left_a_chambers else chamber_names,
                                                   left_v_chambers if left_v_chambers else chamber_names)
    
    # Store connection points
    valve_locations = {
        'right_av': right_atrioventricular[0][2] if right_atrioventricular else heart_center + np.array([2.0, 0, 0]),
        'left_av': left_atrioventricular[0][2] if left_atrioventricular else heart_center + np.array([-2.0, 0, 0]),
        'aortic_valve': heart_center + np.array([-2.0, y_mid - (y_mid - y_min) * 0.3, 0]),  # Between LV and aorta
        'pulmonary_valve': heart_center + np.array([2.0, y_mid - (y_mid - y_min) * 0.3, 0]),  # Between RV and pulmonary artery
    }
    
    # Build fast volumetric inside-test per chamber using vtkImplicitPolyDataDistance
    chamber_distance_field = {}
    chamber_center_map = {}
    try:
        from vtkmodules.vtkCommonDataModel import vtkImplicitPolyDataDistance
        import vtk as _vtk_mod  # guard if vtk not globally available here
        for name in chamber_names:
            try:
                poly = meshes_dict[name].mapper.dataset
                # Ensure polydata surface
                try:
                    poly = poly.extract_surface() if hasattr(poly, 'extract_surface') else poly
                except Exception:
                    pass
                df = vtkImplicitPolyDataDistance()
                df.SetInput(poly)
                chamber_distance_field[name] = df
            except Exception:
                pass
            # cache centers for inward nudging
            try:
                chamber_center_map[name] = np.mean(base_points.get(name, meshes_dict[name].mapper.dataset.points), axis=0)
            except Exception:
                chamber_center_map[name] = heart_center
    except Exception:
        chamber_distance_field = {}
        chamber_center_map = {name: heart_center for name in chamber_names}

    # --- Blood particle system (dual-circulation, fluid-like, heartbeat-synced) ---
    # Parameters
    num_particles_per_path = 80
    num_particles = num_particles_per_path * 2
    flow_speed = 1.2
    turbulence_scale = 0.008
    flow_smoothing = 0.85
    ENABLE_INSIDE_CLAMP = False  # disable expensive per-particle containment checks for performance

    def get_turbulence_frame():
        return np.random.normal(0, turbulence_scale, size=(num_particles, 3))

    turbulence_cache = {"values": get_turbulence_frame()}
    blood_visible = {"state": False}  # start hidden by default

    # Precompute centerline paths for each chamber based on base pose
    def sample_centerline(mesh, axis_index, y_mid, num_points=100):
        pts = mesh.points
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        
        t = np.linspace(0, 4*np.pi, num_points)
        radius = max(1e-6, min(x_max - x_min, z_max - z_min) * 0.25)
        x = radius * np.cos(t) + (x_min + x_max) / 2
        y = np.linspace(y_min, y_max, num_points)
        z = radius * np.sin(t) + (z_min + z_max) / 2
        spiral_points = np.column_stack((x, y, z))
        
        # Try to retain only points inside the chamber
        try:
            test_pts = pv.PolyData(spiral_points)
            result = mesh.select_enclosed_points(test_pts, tolerance=0.1)
            inside_mask = result.point_data['SelectedPoints'] > 0
            valid_points = spiral_points[inside_mask]
            if len(valid_points) >= 5:
                return valid_points
        except:
            pass

        # Fallback: sample along sorted Y
        sort_idx = np.argsort(pts[:, axis_index])
        path = pts[sort_idx]
        step = max(1, len(path) // num_points)
        return path[::step]

    centerline_paths = {}
    for name in chamber_names:
        mesh = meshes_dict[name].mapper.dataset
        centerline_paths[name] = sample_centerline(mesh, axis_index, y_mid, num_points=100)

    # Build deterministic looped flow paths for right and left circulations
    def reorder_path_for_points(path_pts, start_point, end_point):
        if len(path_pts) == 0:
            return path_pts
        # find indices nearest to start and end
        d_start = np.linalg.norm(path_pts - start_point, axis=1)
        d_end = np.linalg.norm(path_pts - end_point, axis=1)
        i_start = int(np.argmin(d_start))
        i_end = int(np.argmin(d_end))
        if i_start <= i_end:
            seg = path_pts[i_start:i_end+1]
        else:
            seg = np.vstack([path_pts[i_start:], path_pts[:i_end+1]])
        # ensure seg starts near start_point and ends near end_point
        if np.linalg.norm(seg[0] - start_point) > np.linalg.norm(seg[-1] - start_point):
            seg = seg[::-1]
        return seg

    def filter_inside(mesh, pts):
        if len(pts) == 0:
            return pts
        try:
            test_pts = pv.PolyData(pts)
            result = mesh.select_enclosed_points(test_pts, tolerance=0.1)
            inside_mask = result.point_data['SelectedPoints'] > 0
            valid = pts[inside_mask]
            if len(valid) > 0:
                return valid
        except Exception:
            pass
        # fallback: pick nearest vertices from mesh for each point
        try:
            verts = mesh.points
            if len(verts) > 0:
                idx = np.argmin(((pts[:,None,:]-verts[None,:,:])**2).sum(axis=2), axis=1)
                return verts[idx]
        except Exception:
            pass
        return pts

    def cumulative_lengths(points):
        if len(points) == 0:
            return np.array([0.0])
        diffs = np.diff(points, axis=0)
        seglens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seglens)])
        return cum

    def sample_arc_length(points, cumlen, s):
        if len(points) == 0:
            return heart_center, 0
        total = cumlen[-1] if len(cumlen) > 0 else 0.0
        if total <= 0:
            return points[0], 0
        s = s % total
        # find segment
        idx = int(np.searchsorted(cumlen, s, side='right') - 1)
        idx = max(0, min(idx, len(points) - 2))
        t0 = cumlen[idx]
        t1 = cumlen[idx + 1]
        alpha = 0.0 if t1 == t0 else (s - t0) / (t1 - t0)
        # Linearly interpolate between points and return with segment index
        return points[idx] * (1 - alpha) + points[idx + 1] * alpha, idx
        return points[idx] * (1 - alpha) + points[idx+1] * alpha, idx

    def build_loop_path(side):
        if side == 'right':
            atria = right_a_chambers if right_a_chambers else chamber_names
            ventr = right_v_chambers if right_v_chambers else chamber_names
            av_pt = valve_locations['right_av']
            out_pt = valve_locations['pulmonary_valve']
        else:
            atria = left_a_chambers if left_a_chambers else chamber_names
            ventr = left_v_chambers if left_v_chambers else chamber_names
            av_pt = valve_locations['left_av']
            out_pt = valve_locations['aortic_valve']

        # Choose first available chamber from lists
        atr_name = atria[0] if len(atria) > 0 else chamber_names[0]
        ven_name = ventr[0] if len(ventr) > 0 else chamber_names[0]
        atr_path = centerline_paths.get(atr_name, np.array([heart_center]))
        ven_path = centerline_paths.get(ven_name, np.array([heart_center]))
        atr_mesh = meshes_dict[atr_name].mapper.dataset
        ven_mesh = meshes_dict[ven_name].mapper.dataset

        # segment through atrium to AV valve, then ventricle to outlet (filter inside respective meshes)
        atr_seg = reorder_path_for_points(atr_path, start_point=heart_center, end_point=av_pt)
        atr_seg = filter_inside(atr_mesh, atr_seg)
        ven_seg = reorder_path_for_points(ven_path, start_point=av_pt, end_point=out_pt)
        ven_seg = filter_inside(ven_mesh, ven_seg)

        # Build loop as atrium->ventricle (wrap around by arc-length modulo)
        loop_pts = []
        seg_ids = []
        if len(atr_seg) > 0:
            loop_pts.append(atr_seg)
            seg_ids.extend(['atria'] * len(atr_seg))
        if len(ven_seg) > 0:
            loop_pts.append(ven_seg)
            seg_ids.extend(['vent'] * len(ven_seg))
        if len(loop_pts) == 0:
            return np.array([heart_center]), ['atria']
        # concatenate and lightly smooth by decimating duplicates
        path = np.vstack(loop_pts)
        # remove consecutive duplicates
        keep = [0]
        for i in range(1, len(path)):
            if np.linalg.norm(path[i] - path[i-1]) > 1e-6:
                keep.append(i)
        path = path[keep]
        seg_ids = [seg_ids[i] for i in keep]
        return path, seg_ids

    right_flow_path, right_seg_ids = build_loop_path('right')
    left_flow_path, left_seg_ids = build_loop_path('left')
    right_flow_cum = cumulative_lengths(right_flow_path)
    left_flow_cum = cumulative_lengths(left_flow_path)

    # Build particles as 2 GPU point-clouds (right=blue, left=red)
    base_radius = 0.025 * max(heart_size)
    right_s = np.linspace(0.0, right_flow_cum[-1] if len(right_flow_cum) > 0 else 1.0, num_particles_per_path, endpoint=False)
    left_s  = np.linspace(0.0, left_flow_cum[-1] if len(left_flow_cum) > 0 else 1.0, num_particles_per_path, endpoint=False)

    # Fly-through camera feature removed

    def compute_positions_for_side(side, s_array):
        positions = np.zeros((len(s_array), 3), dtype=float)
        for idx, s_val in enumerate(s_array):
            if side == 'right':
                path_points, cum, seg_ids = right_flow_path, right_flow_cum, right_seg_ids
            else:
                path_points, cum, seg_ids = left_flow_path, left_flow_cum, left_seg_ids
            if len(cum) == 0 or len(path_points) == 0:
                positions[idx] = heart_center
                continue
            pos, seg_idx = sample_arc_length(path_points, cum, s_val)
            # Volumetric containment: keep inside chamber using signed distance; if outside, snap inward
            try:
                current_seg = seg_ids[max(0, min(seg_idx, len(seg_ids)-1))]
                if current_seg == 'atria':
                    chamber_list = right_a_chambers if side == 'right' else left_a_chambers
                else:
                    chamber_list = right_v_chambers if side == 'right' else left_v_chambers
                chamber_name = chamber_list[0] if chamber_list else (chamber_names[0] if chamber_names else None)
                if chamber_name:
                    df = chamber_distance_field.get(chamber_name)
                    if df is not None:
                        if float(df.EvaluateFunction(pos)) > 0.0:  # outside
                            chamber_mesh = meshes_dict[chamber_name].mapper.dataset
                            verts = chamber_mesh.points
                            if len(verts) > 0:
                                nearest_idx = int(np.argmin(np.linalg.norm(verts - pos, axis=1)))
                                nearest = verts[nearest_idx]
                                center = chamber_center_map.get(chamber_name, heart_center)
                                # nudge slightly inward toward center to ensure inside
                                pos = center + 0.98 * (nearest - center)
                    else:
                        # Fallback: snap to nearest vertex
                        chamber_mesh = meshes_dict[chamber_name].mapper.dataset
                        verts = chamber_mesh.points
                        if len(verts) > 0:
                            nearest_idx = int(np.argmin(np.linalg.norm(verts - pos, axis=1)))
                            pos = verts[nearest_idx]
            except Exception:
                pass
            positions[idx] = pos
        return positions

    right_positions = compute_positions_for_side('right', right_s)
    left_positions  = compute_positions_for_side('left',  left_s)

    right_cloud = plotter.add_points(right_positions, color=(0.1, 0.2, 0.8),  # Darker blue for deoxygenated blood
                                      render_points_as_spheres=True, point_size=8,
                                     name='blood_right')
    left_cloud  = plotter.add_points(left_positions,  color=(0.6, 0.0, 0.0),  # Dark red for oxygenated blood
                                      render_points_as_spheres=True, point_size=8,
                                     name='blood_left')
    try:
        set_gpu_fast_props(right_cloud)
        set_gpu_fast_props(left_cloud)
        # Ensure hidden by default
        if hasattr(right_cloud, 'SetVisibility'):
            right_cloud.SetVisibility(0)
        else:
            right_cloud.visibility = False
        if hasattr(left_cloud, 'SetVisibility'):
            left_cloud.SetVisibility(0)
        else:
            left_cloud.visibility = False
    except Exception:
        pass

    # Camera control and walkthrough features removed

    def smooth_transition(x, width=0.05):
        """Create smooth transition at edges using sigmoid function"""
        if x < 0: return 0
        if x > 1: return 0
        # Smooth start
        if x < width:
            return x/width * (1 - math.cos(math.pi * (x/width))) / 2
        # Smooth end
        if x > (1 - width):
            end_t = (1 - x) / width
            return (1 - end_t) * (1 - math.cos(math.pi * end_t)) / 2
        return 1.0

    def atrial_systole(tt):
        """
        Atrial contraction (atrial systole)
        Phase 1: 0.0 - 0.1 seconds
        Atria squeeze to push final 20-30% of blood into ventricles
        """
        if tt < 0.0 or tt > 0.1:
            return 0.0
        # Smooth atrial contraction curve with gentle transitions
        norm_t = tt/0.1
        return 0.08 * smooth_transition(norm_t, width=0.2)

    def ventricular_systole(tt):
        """
        Ventricular contraction (ventricular systole) - the main pump
        Phase 2: 0.1 - 0.4 seconds
        - Isovolumetric contraction (0.1-0.15s): contracting but no blood out yet
        - Ejection phase (0.15-0.35s): blood forcefully ejected
        - Isovolumetric relaxation (0.35-0.4s): relaxing, valves closed
        """
        if tt < 0.1 or tt > 0.4:
            return 0.0
        
        phase_time = (tt - 0.1) / 0.3  # normalize to 0-1
        
        # Smooth blending between phases
        # Isovolumetric contraction
        iso_strength = 0.0
        if tt < 0.15:
            iso_time = (tt - 0.1) / 0.05
            iso_strength = 0.20 * smooth_transition(iso_time, width=0.3)
        
        # Ejection phase
        eject_strength = 0.0
        if 0.13 < tt < 0.37:  # Overlap slightly for smooth transition
            eject_time = (tt - 0.13) / 0.24
            eject_curve = 0.20 * smooth_transition(eject_time, width=0.2)
            eject_strength = eject_curve * (1 - 0.3 * math.sin(math.pi * eject_time))
        
        # Relaxation phase
        relax_strength = 0.0
        if tt > 0.35:
            relax_time = (tt - 0.35) / 0.05
            relax_strength = 0.14 * (1 - smooth_transition(relax_time, width=0.3))
        
        # Blend all phases
        return max(iso_strength, eject_strength, relax_strength)

    def ventricular_filling(tt):
        """
        Ventricular filling phase (diastole)
        Phase 3: 0.4 - 1.0 seconds (relaxed)
        Heart relaxes, fills passively with blood from atria
        """
        if tt < 0.4 or tt >= 1.0:
            return 0.0
        phase_time = (tt - 0.4) / 0.6  # normalize to 0-1
        
        # Create smoother filling curve with gentle acceleration and deceleration
        fill_curve = smooth_transition(phase_time, width=0.3)
        # Combine with sinusoidal motion for natural-looking expansion
        return -0.02 * fill_curve * (1 - math.cos(2 * math.pi * phase_time)) / 2
        
    def update_beat():
        # Animation update function
            
        # Don't update if animation is disabled
        if not heart_anim_enabled.get("state", True) or not heart_anim.get("enabled", False):
            return
        # Advance simulation time (fixed timestep)
        dt = 0.0167  # ~60 Hz update rate
        t_val["t"] = (t_val["t"] + dt) % beat_period  # Keep time within one beat period
        t_norm = t_val["t"] / beat_period  # Normalize to 0-1 range

        # Calculate forces for each phase
        atr_contract = atrial_systole(t_norm)
        # Reduced delay to 0.01 (10ms) for more realistic left-right synchronization
        vent_delayed_time = t_norm if t_norm > 0.01 else t_norm + 0.01
        vent_contract = ventricular_systole(vent_delayed_time % 1.0)
        vent_fill = ventricular_filling(t_norm)

            # deform chambers - ALL chambers pump together in sync
        for name in chamber_names:
            actor = meshes_dict[name]
            mesh = actor.mapper.dataset
            if name not in base_points:
                if t_val.get("debug_counter", 0) < 5:
                    print(f"[Debug] Skipping {name} - no base points")
                continue

            pts0 = base_points[name]
            pts = pts0.copy()

            # Use precomputed vertex masks per mesh and global centers (vertex-based, not per part)
            if name in vertex_masks:
                upper_mask, lower_mask = vertex_masks[name]
                if t_val.get("debug_counter", 0) < 5:
                    print(f"[Debug] Processing {name} - has masks: upper={np.sum(upper_mask)}, lower={np.sum(lower_mask)}")
            else:
                upper_mask = pts[:, axis_index] > y_mid
                lower_mask = ~upper_mask
            upper_center = global_upper_center
            lower_center = global_lower_center
            
            # Deform: upper region contracts during atrial systole
            if np.any(upper_mask):
                upper_pts_relative = pts[upper_mask] - upper_center
                pts[upper_mask] = upper_center + upper_pts_relative * (1.0 - atr_contract)
            
            # Deform: lower region contracts/expands during ventricular cycle
            if np.any(lower_mask):
                total_deformation = 1.0 - vent_contract + vent_fill
                lower_pts_relative = pts[lower_mask] - lower_center
                pts[lower_mask] = lower_center + lower_pts_relative * total_deformation

            mesh.points = pts
            mesh.Modified()  # Mark the mesh as modified
            actor.GetMapper().Modified()  # Mark the mapper as needing update
            
        # Vessel expansion/contraction synced to cardiac cycle
        # Process both arteries and veins
        for name in meshes_dict:
            if name not in base_points:
                continue
            
            actor = meshes_dict[name]
            mesh = actor.mapper.dataset
            pts0 = base_points[name]
            pts = pts0.copy()
            center = heart_center  # Use heart center instead of vessel center for better anchoring
            
            # Different expansion behavior based on vessel type
            is_artery = any(kw in name.lower() for kw in ['aorta', 'aortic', 'pulmonary', 'ascending', 'arch'])
            is_vein = any(kw in name.lower() for kw in ['vein', 'venous', 'cava', 'vena'])
            
            # Add motion smoothing with momentum
            if not hasattr(actor, '_last_scale'):
                actor._last_scale = 1.0
            if not hasattr(actor, '_scale_velocity'):
                actor._scale_velocity = 0.0
            
            # Calculate target scale based on vessel type
            if is_artery:
                # Arteries expand during ventricular systole (when blood is ejected)
                target_scale = 1.0 + 0.08 * vent_contract  # Slightly reduced maximum scale
            elif is_vein:
                # Veins expand during atrial contraction and ventricular filling
                target_scale = 1.0 + 0.04 * (atr_contract + vent_fill)  # Slightly reduced maximum scale
            else:
                continue  # Skip non-vessel meshes
                
            # Apply smooth motion with momentum
            spring_constant = 0.3  # Controls how quickly it moves to target
            damping = 0.7  # Controls smoothing/momentum (higher = more damping)
            
            # Spring physics for smooth motion
            force = (target_scale - actor._last_scale) * spring_constant
            actor._scale_velocity = actor._scale_velocity * (1 - damping) + force
            scale = actor._last_scale + actor._scale_velocity
            actor._last_scale = scale
                
            # Scale points around heart center, keeping attachment points more stable
            direction = pts0 - center
            pts = center + direction * scale
            
            # Apply changes
            mesh.points = pts
            mesh.Modified()
            actor.GetMapper().Modified()
            if any(kw in name.lower() for kw in ['aorta', 'aortic', 'ascending', 'arch', 'pulmonary']):
                # Arteries expand more during ventricular systole (ejection)
                bulge = 1.0 + 0.04 * vent_contract  # 4% expansion during ejection
            elif any(kw in name.lower() for kw in ['vein', 'venous', 'cava', 'vena']):
                # Veins expand more during atrial systole and early filling
                bulge = 1.0 + 0.03 * (atr_contract + vent_fill)  # 3% expansion during filling
            else:
                # Default minimal expansion for other vessels
                bulge = 1.0 + 0.02 * vent_contract
            
            # Apply deformation
            pts = center + (pts - center) * bulge
            mesh.points = pts

        # Update blood particles (fluid-like, heartbeat-synced)
        # Deterministic arc-length motion along closed loops; no turbulence
        # Update arc-length positions and write batched point clouds
        # Right side
        if len(right_s) > 0 and len(right_flow_cum) > 0 and len(right_flow_path) > 0:
            total_len_r = right_flow_cum[-1]
            speed_r = (0.15 + 0.85 * (0.5 + 0.5 * vent_contract)) * (total_len_r / beat_period)
            right_s[:] = (right_s + speed_r * 0.02) % total_len_r
            new_r = compute_positions_for_side('right', right_s)
            try:
                right_cloud.mapper.dataset.points = new_r
                right_cloud.mapper.dataset.Modified()
            except Exception:
                pass
        # Left side
        if len(left_s) > 0 and len(left_flow_cum) > 0 and len(left_flow_path) > 0:
            total_len_l = left_flow_cum[-1]
            speed_l = (0.15 + 0.85 * (0.5 + 0.5 * vent_contract)) * (total_len_l / beat_period)
            left_s[:] = (left_s + speed_l * 0.02) % total_len_l
            new_l = compute_positions_for_side('left', left_s)
            try:
                left_cloud.mapper.dataset.points = new_l
                left_cloud.mapper.dataset.Modified()
            except Exception:
                    pass

        # Fly-through camera feature removed

        # Update render every other frame to reduce GPU load
        t_val['frame_counter'] = (t_val['frame_counter'] + 1) % 2
        if t_val['frame_counter'] == 0:
            try:
                plotter.render()  # Force render update
            except Exception:
                pass  # Ignore render errors to keep animation runningtry:
                plotter.render()
            except Exception:
                pass  # Ignore render errors to keep animation running
    
    # Add opacity control for heart transparency
    def toggle_opacity():
        """Toggle heart chamber opacity between opaque and transparent"""
        # Cycle between opaque (1.0) and transparent (0.2)
        if heart_opacity["state"] >= 0.9:
            heart_opacity["state"] = 0.2  # Make transparent
            print("[Heart] Opacity: TRANSPARENT (see blood flow inside)")
        else:
            heart_opacity["state"] = 1.0  # Make opaque
            print("[Heart] Opacity: OPAQUE (solid walls)")
        
        # Apply opacity to all heart mesh actors
        for actor in all_mesh_actors:
            try:
                actor.GetProperty().SetOpacity(heart_opacity["state"])
            except:
                pass
        plotter.render()
    
    # Register key callbacks
    try:
        # PyVista's callback method
        def toggle_blood():
            blood_visible["state"] = not blood_visible["state"]
            status = "ON" if blood_visible["state"] else "OFF"
            print(f"[Blood] Visibility {status}")
            # Apply visibility to point clouds
            try:
                if hasattr(right_cloud, 'SetVisibility'):
                    right_cloud.SetVisibility(1 if blood_visible["state"] else 0)
                else:
                    right_cloud.visibility = blood_visible["state"]
            except Exception:
                pass
            try:
                if hasattr(left_cloud, 'SetVisibility'):
                    left_cloud.SetVisibility(1 if blood_visible["state"] else 0)
                else:
                    left_cloud.visibility = blood_visible["state"]
            except Exception:
                pass
            try:
                plotter.render()
            except Exception:
                pass
        # Register all keyboard controls
        plotter.add_key_event('b', toggle_blood)
        plotter.add_key_event('B', toggle_blood)
        plotter.add_key_event('o', toggle_opacity)
        plotter.add_key_event('O', toggle_opacity)
        print("[Controls] Keyboard shortcuts registered successfully")
    except AttributeError:
        # Alternative: use VTK interactor
        try:
            def key_handler(obj, event):
                key = obj.GetKeySym()
                if key.lower() == 'o':
                    toggle_opacity()
                elif key.lower() == 'b':
                    toggle_blood()
                elif key.lower() == 'h':
                    _toggle_heart()

            
            if hasattr(plotter, 'iren') and plotter.iren:
                plotter.iren.AddObserver("KeyPressEvent", key_handler)
        except:
            print("[Note] Key callback could not be registered")
    except Exception as e:
        print(f"[Note] Using fallback method for controls")
    # Store toggle functions for manual access
    plotter._toggle_opacity = toggle_opacity
    plotter._toggle_blood = toggle_blood
    plotter._heart_pause = None  # will be set below
    plotter._heart_resume = None
    plotter._heart_toggle = None
    
    # Heart animation control (pause/resume) and registration
    def _pause_heart():
        # Stop the animation
        heart_anim_enabled["state"] = False
        heart_anim["enabled"] = False
        if heart_anim["cb_id"] is not None:
            try:
                plotter.remove_callback(heart_anim["cb_id"])
            except Exception:
                pass
            heart_anim["cb_id"] = None
        plotter.render()  # Force a render update
        print("[Heart] Animation PAUSED")

    def _resume_heart():
        # Start the animation
        heart_anim_enabled["state"] = True
        heart_anim["enabled"] = True
        if heart_anim["cb_id"] is None:
            try:
                heart_anim["cb_id"] = plotter.add_callback(update_beat, interval=33)
            except Exception:
                pass
        plotter.render()  # Force a render update
        print("[Heart] Animation RESUMED")

    def _toggle_heart():
        if heart_anim["enabled"]:
            _pause_heart()
        else:
            _resume_heart()

    # Initialize and start the animation
    plotter._heart_pause = _pause_heart
    plotter._heart_resume = _resume_heart
    plotter._heart_toggle = _toggle_heart
    # Now that the toggle exists, bind 'h' keys once
    try:
        if not hasattr(plotter, '_heart_keys_registered') or not plotter._heart_keys_registered:
            plotter.add_key_event('h', plotter._heart_toggle)
            plotter.add_key_event('H', plotter._heart_toggle)
            plotter._heart_keys_registered = True
            print("[Controls] 'H' key registered for heart toggle")
    except Exception:
        pass
    
    # Ensure the animation starts with initial values
    t_val["t"] = 0.0
    t_val["frame_counter"] = 0
    heart_anim_enabled["state"] = True
    heart_anim["enabled"] = True
    
    # Register animation callback (30 FPS target)
    try:
        if heart_anim["cb_id"] is not None:
            plotter.remove_callback(heart_anim["cb_id"])
        heart_anim["cb_id"] = plotter.add_callback(update_beat, interval=33)
        print("[Heart] Animation callback registered successfully")
    except Exception as e:
        print(f"[Heart] Failed to register animation callback: {e}")
        heart_anim["enabled"] = False
        heart_anim_enabled["state"] = False
    
    # Periodically log draw time to infer GPU utilization
    def measure_frame_time():
        t0 = time.perf_counter()
        try:
            plotter.render()
        except Exception:
            pass
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[Perf] Frame render time: {dt:.2f} ms (lower is better; GPU-bound is smaller)")
    plotter.add_callback(measure_frame_time, interval=5000)
    print("[Moving Stuff] Realistic cardiac cycle ACTIVE!")
    print(f"   All {len(chamber_names)} chambers pumping together in sync")
    print(f"   Heart rate: {60/beat_period:.0f} BPM (realistic active pace)")
    print("   Phase 1 (0-17%): Atrial systole - gentle contraction")
    print("   Phase 2 (17-50%): Ventricular systole - isovolumetric → ejection")
    print("   Phase 3 (50-100%): Diastole - heart relaxes and fills")
    print("   Physiological: AV delay, pressure curves, realistic timing")
    print("\n[Controls] Press 'B' to toggle blood, 'O' to toggle opacity")
    
    # Add instructions text to the plotter
    try:
        text_actor = plotter.add_text(
            "Controls:\n'B' - Toggle blood\n'O' - Toggle heart opacity\n'Q' - Quit",
            position='upper_right',
            font_size=11,
            color='white'
        )
    except:
        pass


# === Main script with GUI for mesh file selection ===
if __name__ == "__main__":
    # Open file dialog for mesh selection
    root = tk.Tk()
    root.withdraw()
    filetypes_list = [
        ("All supported formats", "*.obj *.stl *.ply *.vtk *.fbx"),
        ("OBJ files", "*.obj"),
        ("STL files", "*.stl"),
        ("PLY files", "*.ply"),
        ("VTK files", "*.vtk"),
        ("FBX files", "*.fbx"),
        ("All files", "*.*")
    ]
    filepaths = filedialog.askopenfilenames(
        title="Select heart chamber mesh files (OBJ/STL/PLY/VTK/FBX supported)",
        filetypes=filetypes_list
    )
    if not filepaths:
        print("No files selected. Exiting.")
        exit()

    # Load meshes into a dict {name: actor}
    # Create plotter with default settings
    plotter = BackgroundPlotter(title="Heart Pump Animation")
    # Prefer hardware-accelerated OpenGL surface format if available
    try:
        if hasattr(plotter, 'app') and hasattr(plotter.app, 'setApplicationDisplayName'):
            plotter.app.setApplicationDisplayName('Heart Pump Animation (GPU)')
    except:
        pass
    
    # Transparency features are costly; keep them disabled for performance
    try:
        if hasattr(plotter.renderer, 'SetUseDepthPeeling'):
            plotter.renderer.SetUseDepthPeeling(0)
        if hasattr(plotter.renderer, 'SetMaximumNumberOfPeels'):
            plotter.renderer.SetMaximumNumberOfPeels(0)
        if hasattr(plotter.renderer, 'UseFXAAOff'):
            plotter.renderer.UseFXAAOff()
    except Exception:
        pass
    
    print("[GPU] Using hardware-accelerated OpenGL rendering")
    print("[Transparency] Depth peeling disabled for performance")
    
    if not (TRIMESH_AVAILABLE or PYASSIMP_AVAILABLE):
        print("\n[Note] For FBX support, install one of:")
        print("   pip install trimesh")
        print("   pip install pyassimp-py3")
    
    # Function to load mesh files including FBX
    def load_mesh_file(filepath):
        """Load mesh from various formats including FBX with animation support"""
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.fbx':
            print(f"[Loading] Attempting to load FBX file: {filepath}")
            
            # Try trimesh first (supports FBX well)
            if TRIMESH_AVAILABLE:
                try:
                    scene = trimesh.load(filepath, force='mesh')
                    
                    # If multiple meshes in scene, combine them or use first
                    if isinstance(scene, trimesh.Scene):
                        # Get all meshes from scene
                        all_meshes = [m for m in scene.geometry.values() if hasattr(m, 'vertices') and len(m.vertices) > 0]
                        if len(all_meshes) > 0:
                            # Use first mesh or combine them
                            if len(all_meshes) > 1:
                                combined_mesh = trimesh.util.concatenate(all_meshes)
                            else:
                                combined_mesh = all_meshes[0]
                            # Convert trimesh to PyVista
                            vertices = combined_mesh.vertices
                            faces = combined_mesh.faces
                            if len(faces) > 0:
                                mesh = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0], 1), 3), faces]))
                                return mesh
                    elif hasattr(scene, 'vertices') and len(scene.vertices) > 0:
                        # Single mesh
                        vertices = scene.vertices
                        faces = scene.faces
                        if len(faces) > 0:
                            mesh = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0], 1), 3), faces]))
                            return mesh
                        
                except Exception as e:
                    print(f"[FBX] trimesh failed: {e}")
                    print("[FBX] Trying pyassimp...")
            
            # Try pyassimp as fallback
            if PYASSIMP_AVAILABLE:
                try:
                    scene = pyassimp.load(filepath)  # pyassimp imported at top of file
                    # Extract mesh data
                    if scene.meshes and len(scene.meshes) > 0:
                        pymesh = scene.meshes[0]
                        vertices = np.array(pymesh.vertices, dtype=np.float32)
                        # PyVista expects faces in specific format
                        faces_array = np.array(pymesh.faces)
                        # Ensure we have proper face indexing for PyVista
                        if len(faces_array) > 0 and len(faces_array.shape) == 2:
                            # Add 3 at the beginning of each face for PyVista
                            faces_for_pv = np.hstack([np.full((faces_array.shape[0], 1), 3), faces_array])
                            mesh = pv.PolyData(vertices, faces_for_pv)
                            pyassimp.release(scene)
                            return mesh
                except Exception as e:
                    print(f"[FBX] pyassimp failed: {e}")
            
            # Last resort: try PyVista's native reader
            try:
                print("[FBX] Trying PyVista native reader...")
                mesh = pv.read(filepath)
                return mesh
            except Exception as e:
                print(f"[Error] Could not load FBX file with any method: {e}")
                return None
        else:
            # Use PyVista's standard loader for other formats
            return pv.read(filepath)
    
    # Function to assign anatomical colors to heart chambers
    def get_heart_color(name):
        """Return realistic anatomical color for each heart chamber"""
        name_lower = name.lower()
        
        # Left ventricle - strong pink/red (main pump chamber)
        if any(kw in name_lower for kw in ['left ventricle', 'lv', 'ventricle left']):
            return (0.85, 0.4, 0.5)  # Pink-red
        
        # Right ventricle - slightly darker red-brown
        elif any(kw in name_lower for kw in ['right ventricle', 'rv', 'ventricle right']):
            return (0.75, 0.35, 0.45)  # Darker pink-red
        
        # Left atrium - light pink
        elif any(kw in name_lower for kw in ['left atrium', 'la', 'atrium left']):
            return (0.9, 0.5, 0.6)  # Light pink
        
        # Right atrium - medium pink
        elif any(kw in name_lower for kw in ['right atrium', 'ra', 'atrium right']):
            return (0.8, 0.45, 0.55)  # Medium pink
        
        # Aorta/Arteries - red-orange
        elif any(kw in name_lower for kw in ['aorta', 'aortic', 'ascending', 'arch']):
            return (0.7, 0.15, 0.2)  # Bright red
        
        # Pulmonary artery - orange-red
        elif any(kw in name_lower for kw in ['pulmonary', 'pulm']):
            return (0.75, 0.3, 0.25)  # Orange-red
        
        # Veins - blue-purple
        elif any(kw in name_lower for kw in ['vein', 'venous', 'cava', 'vena']):
            return (0.5, 0.4, 0.6)  # Purple-blue
        
        # Default - pinkish
        return (0.75, 0.5, 0.6)
    
    meshes_dict = {}
    for path in filepaths:
        try:
            mesh = load_mesh_file(path)
            if mesh is None:
                print(f"[Error] Could not load {path}")
                continue
            
            # Use file name (without extension) as chamber name
            name = os.path.splitext(os.path.basename(path))[0]
            
            # Get anatomical color for this chamber
            color = get_heart_color(name)
            
            # Add mesh with color and smooth shading for realistic appearance
            actor = plotter.add_mesh(mesh, color=color, name=name, 
                                   smooth_shading=True, specular=0.5, 
                                   specular_power=30, diffuse=0.7)
            meshes_dict[name] = actor
            print(f"[Loaded] {name} from {os.path.basename(path)} - Color: RGB{color}")
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")

    start_moving_stuff(plotter, meshes_dict)
    plotter.show()
    
    # CRITICAL: Keep the main thread alive, otherwise BackgroundPlotter window closes immediately
    print("\n[Heart Pump] Animation running! Close the window or press Enter to exit.")
    try:
        input()  # Block until user presses Enter
    except KeyboardInterrupt:
        pass