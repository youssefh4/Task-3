"""3D slice visualization in the 3D scene."""

import numpy as np
from vispy import scene
from slices.viewer import compose_slice_rgb


def visualize_slices_3d(viewer_instance):
    """Create or update 3D slice images perpendicular to each other."""
    if viewer_instance.nifti_data is None:
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.warning(
            viewer_instance, "No NIFTI Loaded",
            "Please load a NIFTI file first."
        )
        return
    
    # Ensure we have bounds even without meshes
    _ensure_bounds_for_slices(viewer_instance)
    
    if viewer_instance.scene_bounds is None:
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.warning(
            viewer_instance, "No Bounds",
            "Unable to determine scene bounds for slices."
        )
        return
    
    # Create visuals if not present
    for view_name in ['axial', 'sagittal', 'coronal']:
        if (view_name not in viewer_instance.slice_images_3d or
            viewer_instance.slice_images_3d[view_name] is None):
            img = scene.visuals.Image(
                np.zeros((10, 10, 3), dtype=np.float32),
                parent=viewer_instance.view.scene,
                clim=(0, 1)
            )
            viewer_instance.slice_images_3d[view_name] = img
    
    viewer_instance.showing_slices_3d = True
    
    # Initial data load and transform
    _update_slices_3d_images(viewer_instance)
    
    # If camera has no initial state and there are no meshes, frame the NIFTI volume
    if (viewer_instance.initial_camera_state is None and
        viewer_instance.compute_bounds() is None and
        viewer_instance.scene_bounds is not None):
        mins, maxs = viewer_instance.scene_bounds
        viewer_instance.view.camera.set_range(
            x=(mins[0], maxs[0]), y=(mins[1], maxs[1]), z=(mins[2], maxs[2])
        )


def _ensure_bounds_for_slices(viewer_instance):
    """Ensure scene_bounds exists; if no meshes, derive from NIFTI volume size."""
    if viewer_instance.scene_bounds is not None:
        return
    
    # Try mesh bounds first
    bounds = viewer_instance.compute_bounds()
    if bounds is not None:
        viewer_instance.scene_bounds = bounds
        return
    
    # Fallback to NIFTI bounds
    if viewer_instance.nifti_shape is not None:
        mins = np.array([0.0, 0.0, 0.0], dtype=float)
        maxs = np.array([
            float(max(0, viewer_instance.nifti_shape[0] - 1)),
            float(max(0, viewer_instance.nifti_shape[1] - 1)),
            float(max(0, viewer_instance.nifti_shape[2] - 1)),
        ], dtype=float)
        viewer_instance.scene_bounds = np.array([mins, maxs])


def update_slices_3d_images(viewer_instance, view_name=None):
    """Update image data for 3D slice images.
    
    Args:
        viewer_instance: STLViewer instance
        view_name: If provided, only update this specific view. 
                   Otherwise update all views.
    """
    if not viewer_instance.showing_slices_3d:
        return
    
    views_to_update = [view_name] if view_name else ['axial', 'sagittal', 'coronal']
    for vname in views_to_update:
        img = viewer_instance.slice_images_3d.get(vname)
        if img is None:
            continue
        
        idx = viewer_instance.slice_sliders[vname].value()
        data = compose_slice_rgb(viewer_instance, vname, idx)
        if data is None:
            continue
        
        img.set_data(data)
        # Store shape (h, w) for transform updates
        viewer_instance.slice_image_shapes[vname] = data.shape[:2]
    
    # Only update transforms if we updated a specific view
    if view_name:
        _update_slices_3d_transforms(viewer_instance)


def _update_slices_3d_images(viewer_instance):
    """Internal helper to update all 3D slice images."""
    update_slices_3d_images(viewer_instance, view_name=None)
    _update_slices_3d_transforms(viewer_instance)


def _update_slices_3d_transforms(viewer_instance):
    """Update transforms of 3D slice images to match current slider indices."""
    if not viewer_instance.showing_slices_3d or viewer_instance.scene_bounds is None:
        return
    
    mins, maxs = viewer_instance.scene_bounds
    size = maxs - mins
    
    # Indices
    axial_idx = viewer_instance.slice_sliders['axial'].value()
    sagittal_idx = viewer_instance.slice_sliders['sagittal'].value()
    coronal_idx = viewer_instance.slice_sliders['coronal'].value()
    
    # Normalized positions
    z_pos = mins[2] + (axial_idx / max(1, (viewer_instance.nifti_shape[2] - 1))) * size[2]
    x_pos = mins[0] + (sagittal_idx / max(1, (viewer_instance.nifti_shape[0] - 1))) * size[0]
    y_pos = mins[1] + (coronal_idx / max(1, (viewer_instance.nifti_shape[1] - 1))) * size[1]
    
    # Axial image transform (XY at z)
    img_ax = viewer_instance.slice_images_3d.get('axial')
    if img_ax is not None and 'axial' in viewer_instance.slice_image_shapes:
        h, w = viewer_instance.slice_image_shapes['axial']
        su = size[0] / max(1, w)
        sv = size[1] / max(1, h)
        img_ax.transform = scene.transforms.STTransform(
            scale=(su, sv, 1.0),
            translate=(mins[0], mins[1], z_pos)
        )
    
    # Coronal image transform (XZ at y) - rotate +90deg about X
    img_co = viewer_instance.slice_images_3d.get('coronal')
    if img_co is not None and 'coronal' in viewer_instance.slice_image_shapes:
        h, w = viewer_instance.slice_image_shapes['coronal']
        su = size[0] / max(1, w)
        sv = size[2] / max(1, h)
        transform = scene.transforms.STTransform(
            scale=(su, sv, 1.0),
            translate=(mins[0], y_pos, mins[2])
        )
        # Apply +90 deg around X axis
        rot = scene.transforms.MatrixTransform()
        rot.rotate(90, (1, 0, 0))
        img_co.transform = transform * rot
    
    # Sagittal image transform (YZ at x) - rotate -90deg about Y
    img_sa = viewer_instance.slice_images_3d.get('sagittal')
    if img_sa is not None and 'sagittal' in viewer_instance.slice_image_shapes:
        h, w = viewer_instance.slice_image_shapes['sagittal']
        su = size[1] / max(1, w)
        sv = size[2] / max(1, h)
        transform = scene.transforms.STTransform(
            scale=(su, sv, 1.0),
            translate=(x_pos, mins[1], mins[2])
        )
        # Apply -90 deg around Y axis
        rot = scene.transforms.MatrixTransform()
        rot.rotate(-90, (0, 1, 0))
        img_sa.transform = transform * rot

