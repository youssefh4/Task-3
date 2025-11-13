"""3D scene management and mesh visualization."""

import numpy as np
from vispy import scene
from vispy.scene import visuals
from vispy.visuals.filters.clipping_planes import PlanesClipper
from PyQt5 import QtCore


def clear_scene(viewer_instance):
    """Safely remove visuals but keep the camera alive."""
    for child in list(viewer_instance.view.scene.children):
        if not isinstance(child, scene.cameras.BaseCamera):
            child.parent = None
    
    viewer_instance.visuals.clear()
    viewer_instance.mesh_clippers.clear()
    for v in list(viewer_instance.plane_visuals.values()):
        v.parent = None
    viewer_instance.plane_visuals.clear()


def compute_bounds(viewer_instance):
    """Compute bounds of checked meshes."""
    bounds = []
    for i in range(viewer_instance.model_list.count()):
        item = viewer_instance.model_list.item(i)
        if item.checkState() == QtCore.Qt.Checked:
            bounds.append(viewer_instance.meshes[item.text()].bounds)
    
    if not bounds:
        return None
    
    mins = np.min([b[0] for b in bounds], axis=0)
    maxs = np.max([b[1] for b in bounds], axis=0)
    return np.array([mins, maxs])


def update_scene(viewer_instance):
    """Rebuild 3D scene based on selected models and opacity."""
    clear_scene(viewer_instance)
    opacity = viewer_instance.model_controls['opacity_slider'].value() / 100.0
    
    # Determine focus state from dropdown BEFORE creating meshes
    selected_model = viewer_instance.model_controls['dropdown'].currentText()
    if selected_model and selected_model != "None" and selected_model in viewer_instance.meshes:
        viewer_instance.focused_model = selected_model
    else:
        viewer_instance.focused_model = None
    
    # Compute scene bounds
    viewer_instance.scene_bounds = compute_bounds(viewer_instance)
    
    # Create mesh visuals for checked items
    for i in range(viewer_instance.model_list.count()):
        item = viewer_instance.model_list.item(i)
        if item.checkState() == QtCore.Qt.Checked:
            mesh = viewer_instance.meshes[item.text()]
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Get or create color for this mesh
            if item.text() not in viewer_instance.mesh_colors:
                viewer_instance.mesh_colors[item.text()] = np.random.rand(3)
            color = viewer_instance.mesh_colors[item.text()]
            
            # Apply focus logic
            if viewer_instance.focused_model is None:
                visual_opacity = opacity
            elif viewer_instance.focused_model == item.text():
                visual_opacity = opacity
            else:
                visual_opacity = opacity * 0.2
            
            mesh_visual = visuals.Mesh(
                vertices=vertices,
                faces=faces,
                color=(color[0], color[1], color[2], visual_opacity),
                shading="smooth",
                parent=viewer_instance.view.scene
            )
            
            # Attach clipping planes filter
            clipper = PlanesClipper(coord_system='scene')
            mesh_visual.attach(clipper)
            viewer_instance.visuals[item.text()] = mesh_visual
            viewer_instance.mesh_clippers[item.text()] = clipper
    
    # Add XYZ axis
    axis = visuals.XYZAxis(parent=viewer_instance.view.scene)
    viewer_instance.view.add(axis)
    
    # Update clipping planes
    from visualization.clipping import get_current_planes
    planes = get_current_planes(viewer_instance)
    for clipper in viewer_instance.mesh_clippers.values():
        clipper.clipping_planes = planes
    
    # Handle camera based on focus
    _update_camera(viewer_instance)
    
    # Store initial camera state if this is the first load
    if (viewer_instance.initial_camera_state is None and
        viewer_instance.scene_bounds is not None):
        mins, maxs = viewer_instance.scene_bounds
        viewer_instance.initial_camera_state = {
            'x_range': (mins[0], maxs[0]),
            'y_range': (mins[1], maxs[1]),
            'z_range': (mins[2], maxs[2])
        }
    
    # If slice images are enabled, recreate/update them
    if viewer_instance.showing_slices_3d:
        from slices.slice_3d import visualize_slices_3d
        visualize_slices_3d(viewer_instance)


def _update_camera(viewer_instance):
    """Update camera position based on focus or scene bounds."""
    if viewer_instance.focused_model is not None:
        # Focus on selected model
        mesh = viewer_instance.meshes[viewer_instance.focused_model]
        mins, maxs = mesh.bounds
        
        viewer_instance.view.camera.set_range(
            x=(mins[0], maxs[0]),
            y=(mins[1], maxs[1]),
            z=(mins[2], maxs[2])
        )
        print(f"Focused on: {viewer_instance.focused_model}")
    elif viewer_instance.scene_bounds is not None:
        # Show all models
        mins, maxs = viewer_instance.scene_bounds
        center = (mins + maxs) / 2
        size = np.max(maxs - mins)
        
        viewer_instance.view.camera.set_range(
            x=(mins[0], maxs[0]),
            y=(mins[1], maxs[1]),
            z=(mins[2], maxs[2])
        )
        print(f"Scene centered at {center}, bounding size {size}")


def reset_camera(viewer_instance):
    """Reset camera to initial state when models first loaded."""
    if viewer_instance.initial_camera_state is None:
        return
    
    # Recreate the camera to reset rotation
    current_fov = viewer_instance.view.camera.fov
    viewer_instance.view.camera = scene.cameras.ArcballCamera(
        fov=current_fov, parent=viewer_instance.view.scene
    )
    
    # Restore initial zoom
    viewer_instance.view.camera.set_range(
        x=viewer_instance.initial_camera_state['x_range'],
        y=viewer_instance.initial_camera_state['y_range'],
        z=viewer_instance.initial_camera_state['z_range']
    )


def clear_views(viewer_instance):
    """Clear all loaded data from views."""
    # Clear 3D model data
    viewer_instance.model_list.clear()
    viewer_instance.model_controls['dropdown'].clear()
    viewer_instance.model_controls['dropdown'].addItem("None")
    viewer_instance.model_controls['dropdown'].setCurrentIndex(0)
    viewer_instance.model_controls['dropdown'].setEnabled(False)
    
    viewer_instance.meshes.clear()
    viewer_instance.mesh_colors.clear()
    if hasattr(viewer_instance, 'mesh_file_paths'):
        viewer_instance.mesh_file_paths.clear()
    clear_scene(viewer_instance)
    viewer_instance.initial_camera_state = None
    viewer_instance.scene_bounds = None
    viewer_instance.focused_model = None
    
    # Clear NIFTI data and slice viewers
    viewer_instance.nifti_data = None
    viewer_instance.nifti_shape = None
    viewer_instance.nifti_affine = None
    
    # Clear mask data
    viewer_instance.mask_data = []
    viewer_instance.mask_files = []
    viewer_instance.mask_colors = []
    viewer_instance._slices_with_masks_cache.clear()
    
    # Clear slice viewers
    from slices.viewer import update_slice_display
    for view_name in viewer_instance.slice_widgets.keys():
        view_widget = viewer_instance.slice_widgets[view_name]
        view = view_widget['view']
        
        # Remove existing image visual
        for child in list(view.scene.children):
            if isinstance(child, scene.visuals.Image):
                child.parent = None
        
        # Reset sliders
        viewer_instance.slice_sliders[view_name].setMinimum(0)
        viewer_instance.slice_sliders[view_name].setMaximum(100)
        viewer_instance.slice_sliders[view_name].setValue(50)
        viewer_instance.slice_labels[view_name].setText(
            f"{view_name.capitalize()} Slice: 0"
        )
    
    print("All views cleared")
    
    # Clear 3D slice images state
    for img in list(viewer_instance.slice_images_3d.values()):
        try:
            img.parent = None
        except Exception:
            pass
    viewer_instance.slice_images_3d = {}
    viewer_instance.showing_slices_3d = False
    viewer_instance.slice_image_shapes = {}
    
    # Clear MPR data
    viewer_instance.marked_points = []
    viewer_instance.mpr_result = None
    viewer_instance.is_marking_points = False

    mark_points_btn = viewer_instance.mpr_controls.get('mark_points')
    generate_btn = viewer_instance.mpr_controls.get('generate')
    reset_btn = viewer_instance.mpr_controls.get('reset')

    if mark_points_btn is not None:
        mark_points_btn.setEnabled(False)
    if generate_btn is not None:
        generate_btn.setEnabled(False)
    if reset_btn is not None:
        reset_btn.setEnabled(False)

    viewer_instance.update_mpr_status()

