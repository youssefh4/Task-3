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
    create_view_controls
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

from gltf.handler import import_gltf_model, import_gltf_animated, WEBENGINE_AVAILABLE


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
            WEBENGINE_AVAILABLE
        )
        # Wire heart animation button
        self.model_controls['heart_animation'].clicked.connect(self.start_heart_animation)
        
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
        
        control_scroll_layout.addStretch()
        control_scroll.setWidget(control_scroll_widget)
        main_layout.insertWidget(0, control_scroll, 1)
        
        # 3D view
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="white")
        self.view = self.canvas.central_widget.add_view()
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
    
    def _connect_signals(self):
        """Connect model dropdown signal."""
        self.model_dropdown.currentIndexChanged.connect(
            lambda: update_scene(self)
        )
        self.model_list.itemChanged.connect(lambda: update_scene(self))
        # Heart button stays enabled; no dynamic state needed
    
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


if __name__ == "__main__":
    import sys
    try:
        app = QtWidgets.QApplication(sys.argv)
        viewer = STLViewer()
        viewer.show()
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        print(f"Error running application: {e}")
        traceback.print_exc()
        sys.exit(1)

