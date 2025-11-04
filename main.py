"""Main application file for 3D Medical Image Viewer."""

import sys
import os
import subprocess
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSettings
from vispy import scene

# Import UI components
from ui.styles import APPLICATION_STYLESHEET
from ui.widgets import (
    create_slice_viewer, create_file_loading_controls,
    create_medical_imaging_controls, create_mpr_controls,
    create_models_controls, create_clipping_controls,
    create_view_controls, create_camera_flythrough_controls
)

# Import handlers
from loaders.file_handlers import (
    load_mesh_folder, load_nifti_file, load_masks_folder,
    visualize_masks_3d
)

from visualization.scene_manager import (
    clear_scene, compute_bounds, update_scene,
    reset_camera, clear_views
)

from visualization.clipping import (
    update_clipping, show_slice_from_clipping
)

from slices.viewer import update_slice_display
from slices.slice_3d import visualize_slices_3d, update_slices_3d_images

from mpr.handlers import (
    toggle_marking_points, generate_mpr, reset_mpr_points,
    update_mpr_status
)

from gltf.handler import import_gltf_model, import_gltf_animated, import_fbx_animated, WEBENGINE_AVAILABLE

from camera.path_manager import (
    record_waypoint, clear_waypoints, generate_orbit_path,
    start_flythrough, stop_flythrough, update_camera_animation,
    get_path_info
)


class STLViewer(QtWidgets.QWidget):
    """Main application window for 3D Medical Image Viewer."""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        self._init_data()
        self._connect_signals()
    
    def _init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("3D Medical Image Viewer - Mesh & Curved MPR Tool")
        self.resize(1400, 900)
        self.settings = QSettings("CMPR", "Viewer")
        self.setStyleSheet(APPLICATION_STYLESHEET)
        
        # Main layout
        main_layout = QtWidgets.QHBoxLayout()
        control_layout = QtWidgets.QVBoxLayout()
        right_column = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_layout, 1)
        main_layout.addLayout(right_column, 3)
        
        # Create slice viewers
        slices_layout = QtWidgets.QHBoxLayout()
        self.slice_widgets = {}
        self.slice_sliders = {}
        self.slice_labels = {}
        
        views = ['axial', 'sagittal', 'coronal']
        for view_name in views:
            slice_container, canvas, slider, label, view = create_slice_viewer(view_name)
            if view_name == 'axial':
                self.axial_canvas = canvas
            self.slice_widgets[view_name] = {
                'canvas': canvas,
                'view': view,
                'image_data': None
            }
            self.slice_sliders[view_name] = slider
            self.slice_labels[view_name] = label
            slices_layout.addLayout(slice_container)
        
        self.setLayout(main_layout)
        
        # Control panel in scrollable area
        control_scroll = QtWidgets.QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setMinimumWidth(280)
        control_scroll.setMaximumWidth(350)
        control_scroll_widget = QtWidgets.QWidget()
        control_scroll_layout = QtWidgets.QVBoxLayout(control_scroll_widget)
        control_scroll_layout.setSpacing(10)
        control_scroll_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create control groups
        self.file_buttons = create_file_loading_controls(
            control_scroll_layout,
            lambda: load_mesh_folder(self),
            lambda: load_nifti_file(self),
            lambda: load_masks_folder(self)
        )
        
        self.medical_buttons = create_medical_imaging_controls(
            control_scroll_layout,
            lambda: visualize_masks_3d(self),
            lambda: visualize_slices_3d(self)
        )
        
        self.mpr_controls = create_mpr_controls(
            control_scroll_layout,
            lambda: toggle_marking_points(self),
            lambda: generate_mpr(self),
            lambda: reset_mpr_points(self)
        )
        
        self.model_controls = create_models_controls(
            control_scroll_layout,
            lambda: import_gltf_model(self),
            lambda: import_gltf_animated(self),
            WEBENGINE_AVAILABLE,
            lambda: import_fbx_animated(self)
        )
        # Wire heart animation button
        self.model_controls['heart_animation'].clicked.connect(self.start_heart_animation)
        # Wire brain electric effect button
        try:
            self.model_controls['brain_electric'].clicked.connect(self.start_brain_electric_effect)
        except Exception:
            pass
        
        # Connect opacity slider
        self.model_controls['opacity_slider'].valueChanged.connect(
            lambda val: self.model_controls['opacity_label'].setText(f"{val}%")
        )
        
        self.clipping_controls = create_clipping_controls(
            control_scroll_layout,
            lambda: update_clipping(self),
            lambda plane_name: show_slice_from_clipping(self, plane_name)
        )
        
        self.view_buttons = create_view_controls(
            control_scroll_layout,
            lambda: update_scene(self),
            lambda: reset_camera(self),
            lambda: clear_views(self)
        )
        
        self.camera_controls = create_camera_flythrough_controls(
            control_scroll_layout,
            lambda: self.record_camera_waypoint(),
            lambda: self.clear_camera_waypoints(),
            lambda: self.start_orbit_path(),
            lambda: self.start_custom_path(),
            lambda: self.stop_camera_animation(),
            lambda: self.pause_camera_animation()
        )
        
        control_scroll_layout.addStretch()
        control_scroll.setWidget(control_scroll_widget)
        main_layout.insertWidget(0, control_scroll, 1)
        
        # 3D view
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="white")
        self.view = self.canvas.central_widget.add_view()
        # Use TurntableCamera for better fly-through support, fallback to ArcballCamera
        try:
            self.view.camera = scene.cameras.TurntableCamera(fov=60)
        except:
            self.view.camera = scene.cameras.ArcballCamera(fov=60)
        right_column.addWidget(self.canvas.native, 3)
        
        # Add slices to right column
        right_column.addLayout(slices_layout, 1)
        
        # Store references for backward compatibility
        self.control_layout = control_layout
        self.right_column = right_column
        self.model_list = self.model_controls['list']
        self.model_dropdown = self.model_controls['dropdown']
        self.opacity_slider = self.model_controls['opacity_slider']
        self.opacity_label = self.model_controls['opacity_label']
        # Ensure heart button is enabled
        try:
            self.model_controls['heart_animation'].setEnabled(True)
        except Exception:
            pass
        # Initialize brain electric availability
        try:
            self._update_brain_effect_enabled()
        except Exception:
            pass
    
    def _init_data(self):
        """Initialize data structures."""
        # Store loaded meshes and visuals
        self.meshes = {}
        self.visuals = {}
        self.mesh_clippers = {}
        self.plane_visuals = {}
        self.initial_camera_state = None
        self.scene_bounds = None
        self.focused_model = None
        self.mesh_colors = {}
        
        # 3D slice images
        self.slice_images_3d = {}
        self.showing_slices_3d = False
        self.slice_image_shapes = {}
        
        # NIFTI data
        self.nifti_data = None
        self.nifti_shape = None
        self.nifti_affine = None
        
        # Mask data
        self.mask_data = []
        self.mask_files = []
        self.mask_colors = []
        
        # Cache for slices with masks
        self._slices_with_masks_cache = {}
        
        # MPR point marking state
        self.is_marking_points = False
        self.marked_points = []
        self.mpr_result = None
        self.mpr_window = None
        
        # GLTF animated viewer windows
        self.gltf_windows = []
        
        # Camera fly-through state
        self.camera_waypoints = []
        self.camera_animation = None
        self.camera_timer = None
    
    def _connect_signals(self):
        """Connect model dropdown signal."""
        self.model_dropdown.currentIndexChanged.connect(
            lambda: update_scene(self)
        )
        self.model_list.itemChanged.connect(lambda: (update_scene(self), self._update_brain_effect_enabled()))
        # Heart button stays enabled; no dynamic state needed
        
        # Setup camera animation timer
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.timeout.connect(self._update_camera_animation)
        self.camera_timer.setInterval(33)  # ~30 FPS
    
    # Wrapper methods for backward compatibility
    def clear_scene(self):
        """Clear 3D scene."""
        clear_scene(self)
    
    def compute_bounds(self):
        """Compute scene bounds."""
        return compute_bounds(self)
    
    def update_scene(self):
        """Update 3D scene."""
        update_scene(self)
        try:
            self.model_controls['heart_animation'].setEnabled(True)
        except Exception:
            pass
    
    def reset_camera(self):
        """Reset camera."""
        reset_camera(self)
    
    def clear_views(self):
        """Clear all views."""
        clear_views(self)
    
    def update_clipping(self):
        """Update clipping planes."""
        update_clipping(self)
    
    def show_slice_from_clipping(self, plane_name):
        """Show slice from clipping plane."""
        show_slice_from_clipping(self, plane_name)
    
    def visualize_slices_3d(self):
        """Visualize slices in 3D."""
        visualize_slices_3d(self)
    
    def update_axial_slice(self, val):
        """Update axial slice."""
        update_slice_display(self, 'axial', val)
        update_slices_3d_images(self, 'axial')
    
    def update_sagittal_slice(self, val):
        """Update sagittal slice."""
        update_slice_display(self, 'sagittal', val)
        update_slices_3d_images(self, 'sagittal')
    
    def update_coronal_slice(self, val):
        """Update coronal slice."""
        update_slice_display(self, 'coronal', val)
        update_slices_3d_images(self, 'coronal')
    
    def toggle_marking_points(self):
        """Toggle MPR point marking."""
        toggle_marking_points(self)
    
    def generate_mpr(self):
        """Generate MPR."""
        generate_mpr(self)
    
    def reset_mpr_points(self):
        """Reset MPR points."""
        reset_mpr_points(self)
    
    def update_mpr_status(self):
        """Update MPR status."""
        update_mpr_status(self)
    
    def import_gltf_model(self):
        """Import GLTF model."""
        import_gltf_model(self)
    
    def import_gltf_animated(self):
        """Import animated GLTF."""
        import_gltf_animated(self)
    
    # Add QtCore access for widgets
    QtCore = QtCore

    # Heart animation integration (launch external script)
    def _has_heart_model(self):
        try:
            for name in self.meshes.keys():
                if 'heart' in str(name).lower():
                    return True
        except Exception:
            pass
        return False

    def _update_heart_button_state(self):
        # Keep the button always enabled as requested
        try:
            self.model_controls['heart_animation'].setEnabled(True)
        except Exception:
            pass

    def start_heart_animation(self):
        # If any loaded model looks like an artery, run animation with loaded meshes in-process
        if self._detects_artery():
            ran = self._run_animation_with_loaded_meshes()
            if ran:
                return
            # If in-process run failed, fall back to launching the script externally
        
        # Fallback: launch external script unchanged
        app_dir, script_path = self._resolve_heart_script_path()
        if not script_path:
            return
        try:
            creationflags = 0
            if sys.platform.startswith('win'):
                creationflags = 0x00000008  # CREATE_NEW_CONSOLE
            subprocess.Popen([sys.executable, script_path], cwd=app_dir, creationflags=creationflags)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Heart Animation",
                f"Failed to launch animation script:\n{e}"
            )

    def _resolve_heart_script_path(self):
        try:
            app_dir = os.path.abspath(os.path.dirname(__file__))
        except Exception:
            app_dir = os.getcwd()
        script_path = os.path.join(app_dir, 'from direct.showbase.py')
        if not os.path.isfile(script_path):
            QtWidgets.QMessageBox.critical(
                self,
                "Heart Animation",
                "Could not find 'from direct.showbase.py' in the project directory."
            )
            return app_dir, None
        return app_dir, script_path

    def _detects_artery(self):
        try:
            artery_keys = ['artery', 'aorta', 'aortic', 'pulmonary']
            for name in self.meshes.keys():
                low = str(name).lower()
                if any(k in low for k in artery_keys):
                    return True
        except Exception:
            pass
        return False

    def _get_heart_color(self, name):
        """Return realistic anatomical color for each heart chamber (same as script)"""
        name_lower = str(name).lower()
        
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

    def _run_animation_with_loaded_meshes(self):
        # Dynamically import the external script without modifying it
        try:
            app_dir, script_path = self._resolve_heart_script_path()
            if not script_path:
                return False
            import importlib.util
            spec = importlib.util.spec_from_file_location("heart_anim_module", script_path)
            if spec is None or spec.loader is None:
                return False
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            # If import fails, let caller fallback
            return False

        # Import pyvista and plotter used by the script
        try:
            import pyvista as pv
            from pyvistaqt import BackgroundPlotter
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Heart Animation",
                "PyVista/PyVistaQt not installed. Install with: pip install pyvista pyvistaqt"
            )
            return False

        # Convert currently checked meshes to PyVista and add to plotter
        try:
            plotter = BackgroundPlotter(title="Heart Pump Animation (Loaded Meshes)")
        except Exception:
            # Fallback to standard plotter
            try:
                plotter = pv.Plotter()
            except Exception:
                return False

        meshes_dict = {}
        try:
            import numpy as _np
            for i in range(self.model_list.count()):
                item = self.model_list.item(i)
                # Only include checked models
                if item.checkState() != QtCore.Qt.Checked:
                    continue
                name = item.text()
                tri = self.meshes.get(name)
                if tri is None:
                    continue
                # Expect trimesh.Trimesh
                try:
                    verts = _np.asarray(tri.vertices)
                    faces = _np.asarray(tri.faces)
                    if faces.ndim != 2 or faces.shape[1] != 3:
                        continue
                    faces_for_pv = _np.hstack([_np.full((faces.shape[0], 1), 3), faces]).astype(_np.int64)
                    poly = pv.PolyData(verts, faces_for_pv)
                except Exception:
                    continue
                # Get realistic anatomical color using the same logic as the script
                color = self._get_heart_color(name)
                try:
                    actor = plotter.add_mesh(poly, color=color, name=name, 
                                           smooth_shading=True, specular=0.5, 
                                           specular_power=30, diffuse=0.7)
                    meshes_dict[name] = actor
                except Exception:
                    pass
        except Exception:
            return False

        # Call the provided function to start animation
        try:
            if hasattr(mod, 'start_moving_stuff'):
                mod.start_moving_stuff(plotter, meshes_dict)
                # Show the plotter window (non-blocking if BackgroundPlotter)
                try:
                    plotter.show()
                except Exception:
                    pass
                return True
        except Exception:
            return False

        return False
    
    # Camera fly-through control methods
    def record_camera_waypoint(self):
        """Record current camera position as a waypoint."""
        if record_waypoint(self):
            self._update_camera_status()
            # Enable custom path button if we have at least 2 waypoints
            if len(self.camera_waypoints) >= 2:
                self.camera_controls['start_custom'].setEnabled(True)
    
    def clear_camera_waypoints(self):
        """Clear all recorded camera waypoints."""
        clear_waypoints(self)
        self._update_camera_status()
        self.camera_controls['start_custom'].setEnabled(False)
    
    def start_orbit_path(self):
        """Start orbit fly-through animation."""
        if self.scene_bounds is None:
            QtWidgets.QMessageBox.warning(
                self, "No Scene", 
                "Please load a 3D model first to create an orbit path."
            )
            return
        
        speed = self.camera_controls['speed_slider'].value() / 100.0
        loop = self.camera_controls['loop_checkbox'].isChecked()
        radius_percent = self.camera_controls['orbit_radius_slider'].value()
        
        if start_flythrough(self, path_mode='orbit', loop=loop, speed=speed, radius_percent=radius_percent):
            self.camera_timer.start()
            self._update_camera_controls_state(playing=True)
            self._update_camera_status()
    
    def start_custom_path(self):
        """Start custom path fly-through animation."""
        if len(self.camera_waypoints) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Not Enough Waypoints",
                "Please record at least 2 waypoints before starting a custom path."
            )
            return
        
        speed = self.camera_controls['speed_slider'].value() / 100.0
        loop = self.camera_controls['loop_checkbox'].isChecked()
        
        if start_flythrough(self, path_mode='custom', loop=loop, speed=speed):
            self.camera_timer.start()
            self._update_camera_controls_state(playing=True)
            self._update_camera_status()
    
    def stop_camera_animation(self):
        """Stop camera fly-through animation."""
        stop_flythrough(self)
        if self.camera_timer:
            self.camera_timer.stop()
        self._update_camera_controls_state(playing=False)
        self._update_camera_status()
    
    def pause_camera_animation(self):
        """Pause or resume camera fly-through animation."""
        if hasattr(self, 'camera_animation') and self.camera_animation:
            anim = self.camera_animation
            anim['paused'] = not anim['paused']
            
            if anim['paused']:
                self.camera_controls['pause'].setText("▶️ Resume")
                self.camera_controls['status'].setText("Status: Paused")
            else:
                self.camera_controls['pause'].setText("⏸️ Pause")
                self.camera_controls['status'].setText("Status: Playing")
        else:
            # If paused and no animation, start from current state
            self._update_camera_status()
    
    def _update_camera_animation(self):
        """Update camera animation frame (called by timer)."""
        if hasattr(self, 'camera_animation') and self.camera_animation:
            dt = 0.033  # ~30 FPS (33ms interval)
            still_playing = update_camera_animation(self, dt)
            
            if not still_playing:
                self.stop_camera_animation()
            else:
                # Update speed if changed
                if self.camera_animation:
                    speed = self.camera_controls['speed_slider'].value() / 100.0
                    self.camera_animation['speed'] = speed
                    loop = self.camera_controls['loop_checkbox'].isChecked()
                    self.camera_animation['loop'] = loop
    
    def _update_camera_controls_state(self, playing=False):
        """Update enabled/disabled state of camera controls."""
        if playing:
            self.camera_controls['start_orbit'].setEnabled(False)
            self.camera_controls['start_custom'].setEnabled(False)
            self.camera_controls['pause'].setEnabled(True)
            self.camera_controls['stop'].setEnabled(True)
            self.camera_controls['record'].setEnabled(False)
        else:
            self.camera_controls['start_orbit'].setEnabled(True)
            self.camera_controls['start_custom'].setEnabled(len(self.camera_waypoints) >= 2)
            self.camera_controls['pause'].setEnabled(False)
            self.camera_controls['stop'].setEnabled(False)
            self.camera_controls['record'].setEnabled(True)
            self.camera_controls['pause'].setText("⏸️ Pause")
    
    def _update_camera_status(self):
        """Update camera status label."""
        waypoint_count = len(self.camera_waypoints)
        self.camera_controls['waypoint_count'].setText(f"Waypoints: {waypoint_count}")
        
        if hasattr(self, 'camera_animation') and self.camera_animation:
            anim = self.camera_animation
            if anim['paused']:
                self.camera_controls['status'].setText("Status: Paused")
            else:
                self.camera_controls['status'].setText("Status: Playing")
        else:
            self.camera_controls['status'].setText("Status: Stopped")

    # --- Brain Electric Effect integration ---
    def _has_brain_region(self) -> bool:
        """Return True if any loaded model name contains midbrain or medulla."""
        try:
            for i in range(self.model_list.count()):
                item = self.model_list.item(i)
                name = item.text().lower()
                if ('midbrain' in name) or ('medulla' in name):
                    return True
        except Exception:
            pass
        return False

    def _update_brain_effect_enabled(self):
        try:
            # Always enable the Brain Electric Effect button
            self.model_controls['brain_electric'].setEnabled(True)
        except Exception:
            pass

    def start_brain_electric_effect(self):
        """Launch the VTK brain electric visualization if appropriate regions are present."""
        if not self._has_brain_region():
            QtWidgets.QMessageBox.information(
                self,
                "Brain Electric Effect",
                "Load a model whose name includes 'midbrain' or 'medulla' to enable this effect."
            )
            return
        try:
            script_path = os.path.join(os.path.dirname(__file__), 'run_pygame_demo.py')
            if not os.path.exists(script_path):
                QtWidgets.QMessageBox.warning(self, "Not Found", f"Script not found: {script_path}")
                return
            # Launch as separate process; let the script prompt for files
            subprocess.Popen([sys.executable, script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Brain Electric Effect", f"Failed to start effect:\n{e}")


if __name__ == "__main__":
    import sys
    try:
        app = QtWidgets.QApplication(sys.argv)
        viewer = STLViewer()
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        print(f"Error running application: {e}")
        traceback.print_exc()
        sys.exit(1)

