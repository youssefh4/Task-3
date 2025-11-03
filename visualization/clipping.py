"""Clipping planes functionality."""

import numpy as np
from vispy import scene
from vispy.scene import visuals


def get_current_planes(viewer_instance):
    """Compute current clipping planes from UI state."""
    planes = []
    if viewer_instance.scene_bounds is None:
        return np.zeros((0, 2, 3), dtype=float)
    
    mins, maxs = viewer_instance.scene_bounds
    center = (mins + maxs) / 2.0
    size = maxs - mins
    
    axial_controls = viewer_instance.clipping_controls['axial']
    coronal_controls = viewer_instance.clipping_controls['coronal']
    sagittal_controls = viewer_instance.clipping_controls['sagittal']
    
    if axial_controls['checkbox'].isChecked():
        z = mins[2] + (axial_controls['slider'].value() / 100.0) * size[2]
        planes.append([np.array([center[0], center[1], z]), np.array([0, 0, -1])])
    
    if coronal_controls['checkbox'].isChecked():
        y = mins[1] + (coronal_controls['slider'].value() / 100.0) * size[1]
        planes.append([np.array([center[0], y, center[2]]), np.array([0, -1, 0])])
    
    if sagittal_controls['checkbox'].isChecked():
        x = mins[0] + (sagittal_controls['slider'].value() / 100.0) * size[0]
        planes.append([np.array([x, center[1], center[2]]), np.array([-1, 0, 0])])
    
    return np.array(planes).reshape(-1, 2, 3) if planes else np.zeros((0, 2, 3))


def update_clipping(viewer_instance):
    """Update clipping planes without rebuilding scene."""
    if viewer_instance.scene_bounds is None:
        return
    
    # Update clipper planes
    planes = get_current_planes(viewer_instance)
    for clipper in viewer_instance.mesh_clippers.values():
        clipper.clipping_planes = planes
    
    # Update plane visuals
    _ensure_plane_visuals(viewer_instance)
    viewer_instance.canvas.update()


def _ensure_plane_visuals(viewer_instance):
    """Create or update plane visuals based on current state."""
    if viewer_instance.scene_bounds is None:
        return
    
    mins, maxs = viewer_instance.scene_bounds
    center = (mins + maxs) / 2.0
    size = maxs - mins
    
    # Remove old planes first
    for v in list(viewer_instance.plane_visuals.values()):
        v.parent = None
    viewer_instance.plane_visuals.clear()
    
    # Create plane visuals for active planes
    axial_controls = viewer_instance.clipping_controls['axial']
    coronal_controls = viewer_instance.clipping_controls['coronal']
    sagittal_controls = viewer_instance.clipping_controls['sagittal']
    
    if axial_controls['checkbox'].isChecked():
        z = mins[2] + (axial_controls['slider'].value() / 100.0) * size[2]
        width, height = size[0], size[1]
        width *= 1.5
        height *= 1.5
        plane_visual = visuals.Plane(
            width=width,
            height=height,
            direction='+z',
            color=(1.0, 0.5, 0.5, 0.3),
            parent=viewer_instance.view.scene
        )
        plane_visual.transform = scene.transforms.STTransform(
            translate=(center[0], center[1], z)
        )
        viewer_instance.plane_visuals['axial'] = plane_visual
    
    if coronal_controls['checkbox'].isChecked():
        y = mins[1] + (coronal_controls['slider'].value() / 100.0) * size[1]
        width, height = size[0], size[2]
        width *= 1.5
        height *= 1.5
        plane_visual = visuals.Plane(
            width=width,
            height=height,
            direction='+y',
            color=(0.5, 1.0, 0.5, 0.3),
            parent=viewer_instance.view.scene
        )
        plane_visual.transform = scene.transforms.STTransform(
            translate=(center[0], y, center[2])
        )
        viewer_instance.plane_visuals['coronal'] = plane_visual
    
    if sagittal_controls['checkbox'].isChecked():
        x = mins[0] + (sagittal_controls['slider'].value() / 100.0) * size[0]
        width, height = size[1], size[2]
        width *= 1.5
        height *= 1.5
        plane_visual = visuals.Plane(
            width=width,
            height=height,
            direction='+x',
            color=(0.5, 0.5, 1.0, 0.3),
            parent=viewer_instance.view.scene
        )
        plane_visual.transform = scene.transforms.STTransform(
            translate=(x, center[1], center[2])
        )
        viewer_instance.plane_visuals['sagittal'] = plane_visual


def show_slice_from_clipping(viewer_instance, plane_name):
    """Show the current clipping plane slice in the corresponding 2D slice viewer."""
    if viewer_instance.nifti_data is None:
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.warning(
            viewer_instance, "No NIFTI Data",
            "Please load a NIFTI file first."
        )
        return
    
    if viewer_instance.scene_bounds is None:
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.warning(
            viewer_instance, "No 3D Model",
            "Please load 3D models first."
        )
        return
    
    try:
        # Get the actual 3D position of the clipping plane and map to NIFTI slice index
        mins, maxs = viewer_instance.scene_bounds
        size = maxs - mins
        center = (mins + maxs) / 2.0
        
        # Calculate 3D position of the clipping plane
        controls = viewer_instance.clipping_controls[plane_name]
        slider_value = controls['slider'].value()
        
        dim_index = None
        if plane_name == 'axial':
            dim_index = 2  # Z dimension -> axial slice
        elif plane_name == 'sagittal':
            dim_index = 0  # X dimension -> sagittal slice
        elif plane_name == 'coronal':
            dim_index = 1  # Y dimension -> coronal slice
        else:
            return
        
        # Ensure slider range is correct
        if viewer_instance.nifti_shape is not None:
            slider_max = viewer_instance.nifti_shape[dim_index] - 1
            if slider_max < 0:
                slider_max = 0
            viewer_instance.slice_sliders[plane_name].setMinimum(0)
            viewer_instance.slice_sliders[plane_name].setMaximum(slider_max)
        else:
            return
        
        # Get slices that have masks (if masks are loaded)
        from slices.viewer import get_slices_with_masks
        valid_slices = get_slices_with_masks(viewer_instance, plane_name)
        
        if valid_slices is not None and len(valid_slices) > 0:
            # Only use slices that contain masks
            percentage = slider_value / 100.0
            valid_index = int(np.round(percentage * (len(valid_slices) - 1)))
            valid_index = max(0, min(valid_index, len(valid_slices) - 1))
            slice_index = valid_slices[valid_index]
            print(f"Using mask-filtered slice: {slice_index} "
                  f"(from {len(valid_slices)} slices with masks, "
                  f"position {valid_index}/{len(valid_slices)-1})")
        else:
            # No masks loaded, use all slices
            percentage = slider_value / 100.0
            slice_index = int(np.round(percentage * slider_max))
            slice_index = max(0, min(slice_index, slider_max))
        
        # Update the slider
        viewer_instance.slice_sliders[plane_name].blockSignals(True)
        viewer_instance.slice_sliders[plane_name].setValue(slice_index)
        viewer_instance.slice_sliders[plane_name].blockSignals(False)
        
        # Manually update the slice
        from PyQt5 import QtWidgets
        QtWidgets.QApplication.processEvents()
        from slices.viewer import update_slice_display
        update_slice_display(viewer_instance, plane_name, slice_index)
        
        # Also update 3D slice images if they're showing
        if viewer_instance.showing_slices_3d:
            from slices.slice_3d import update_slices_3d_images
            update_slices_3d_images(viewer_instance, plane_name)
        
        # Force canvas update
        viewer_instance.slice_widgets[plane_name]['canvas'].update()
        
        print(f"Showing {plane_name} slice {slice_index} "
              f"(from {slider_value}% clipping plane, {percentage:.2f} normalized)")
        
    except Exception as e:
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.critical(
            viewer_instance, "Error", f"Failed to show slice:\n{str(e)}"
        )
        import traceback
        traceback.print_exc()

