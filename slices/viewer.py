"""2D slice viewer logic."""

import numpy as np
from vispy import scene


def setup_slice_sliders(viewer_instance):
    """Setup slider ranges based on NIFTI dimensions."""
    # Axial: Z dimension (shape[2])
    viewer_instance.slice_sliders['axial'].setMinimum(0)
    axial_max = max(0, viewer_instance.nifti_shape[2] - 1)
    viewer_instance.slice_sliders['axial'].setMaximum(axial_max)
    
    # Sagittal: X dimension (shape[0])
    viewer_instance.slice_sliders['sagittal'].setMinimum(0)
    sagittal_max = max(0, viewer_instance.nifti_shape[0] - 1)
    viewer_instance.slice_sliders['sagittal'].setMaximum(sagittal_max)
    
    # Coronal: Y dimension (shape[1])
    viewer_instance.slice_sliders['coronal'].setMinimum(0)
    coronal_max = max(0, viewer_instance.nifti_shape[1] - 1)
    viewer_instance.slice_sliders['coronal'].setMaximum(coronal_max)
    
    # Set initial slice positions to middle
    viewer_instance.slice_sliders['axial'].setValue(
        viewer_instance.nifti_shape[2] // 2
    )
    viewer_instance.slice_sliders['sagittal'].setValue(
        viewer_instance.nifti_shape[0] // 2
    )
    viewer_instance.slice_sliders['coronal'].setValue(
        viewer_instance.nifti_shape[1] // 2
    )


def connect_slice_sliders(viewer_instance):
    """Connect slice slider signals to update functions."""
    # Disconnect any existing connections first
    for view_name in ['axial', 'sagittal', 'coronal']:
        try:
            viewer_instance.slice_sliders[view_name].valueChanged.disconnect()
        except:
            pass
    
    # Connect sliders to update functions
    viewer_instance.slice_sliders['axial'].valueChanged.connect(
        lambda val: viewer_instance.update_axial_slice(val)
    )
    viewer_instance.slice_sliders['sagittal'].valueChanged.connect(
        lambda val: viewer_instance.update_sagittal_slice(val)
    )
    viewer_instance.slice_sliders['coronal'].valueChanged.connect(
        lambda val: viewer_instance.update_coronal_slice(val)
    )


def update_slice_display(viewer_instance, view_name, slice_idx):
    """Update the displayed slice for a given view."""
    if viewer_instance.nifti_data is None:
        return
    
    try:
        # Extract slice based on view type
        slice_data = _extract_slice(viewer_instance.nifti_data, view_name, slice_idx)
        
        # Normalize data to 0-1 range
        slice_data = slice_data.astype(np.float32)
        if slice_data.max() > slice_data.min():
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        
        # Convert to grayscale RGB
        slice_data_rgb = np.stack([slice_data, slice_data, slice_data], axis=2)
        
        # Overlay masks if available
        if viewer_instance.mask_data:
            slice_data_rgb = _overlay_masks(
                viewer_instance, view_name, slice_idx, slice_data_rgb
            )
        
        # Overlay marked points on axial slice
        if view_name == 'axial' and viewer_instance.marked_points:
            slice_data_rgb = _overlay_mpr_points(
                viewer_instance, slice_idx, slice_data_rgb
            )
        
        # Display the image
        _display_slice_image(viewer_instance, view_name, slice_data_rgb, slice_idx)
        
    except Exception as e:
        print(f"Error updating {view_name} slice: {e}")
        import traceback
        traceback.print_exc()


def _extract_slice(nifti_data, view_name, slice_idx):
    """Extract slice data for a given view."""
    if view_name == 'axial':
        return nifti_data[:, :, slice_idx].T
    elif view_name == 'sagittal':
        return nifti_data[slice_idx, :, :].T
    elif view_name == 'coronal':
        slice_data = nifti_data[:, slice_idx, :].T
        return np.fliplr(slice_data)
    else:
        return None


def _overlay_masks(viewer_instance, view_name, slice_idx, slice_data_rgb):
    """Overlay mask data on slice image."""
    composite = slice_data_rgb.copy()
    
    for mask, color in zip(viewer_instance.mask_data, viewer_instance.mask_colors):
        # Extract mask slice
        if view_name == 'axial':
            mask_slice = mask[:, :, slice_idx].T
        elif view_name == 'sagittal':
            mask_slice = mask[slice_idx, :, :].T
        elif view_name == 'coronal':
            mask_slice = mask[:, slice_idx, :].T
            mask_slice = np.fliplr(mask_slice)
        else:
            continue
        
        # Create mask overlay - threshold mask
        mask_overlay = mask_slice > 0.5
        
        # Skip if no mask pixels in this slice
        if not np.any(mask_overlay):
            continue
        
        # Blend mask color with opacity
        alpha = 0.5
        composite[mask_overlay] = (
            (1 - alpha) * composite[mask_overlay] + alpha * color
        )
    
    return composite


def _overlay_mpr_points(viewer_instance, slice_idx, slice_data_rgb):
    """Overlay MPR marked points on slice image."""
    for pt in viewer_instance.marked_points:
        if int(pt[2]) == slice_idx:  # Point on current slice
            # Points are stored as [x, y, z] in NIFTI coordinates
            img_x = int(pt[0])  # x in NIFTI = column in displayed
            img_y = int(pt[1])  # y in NIFTI = row in displayed
            # Draw cross marker
            size = 5
            for dy in range(-size, size+1):
                for dx in range(-size, size+1):
                    if abs(dy) == abs(dx) or dy == 0 or dx == 0:
                        py, px = img_y + dy, img_x + dx
                        if (0 <= py < slice_data_rgb.shape[0] and
                            0 <= px < slice_data_rgb.shape[1]):
                            slice_data_rgb[py, px] = [1.0, 1.0, 0.0]  # Yellow
    return slice_data_rgb


def _display_slice_image(viewer_instance, view_name, slice_data_rgb, slice_idx):
    """Display slice image in the viewer."""
    view_widget = viewer_instance.slice_widgets[view_name]
    view = view_widget['view']
    
    # Remove existing image visual if any
    for child in list(view.scene.children):
        if isinstance(child, scene.visuals.Image):
            child.parent = None
    
    # Create new image visual
    image = scene.visuals.Image(slice_data_rgb, parent=view.scene)
    
    # Update label
    viewer_instance.slice_labels[view_name].setText(
        f"{view_name.capitalize()} Slice: "
        f"{slice_idx}/{viewer_instance.slice_sliders[view_name].maximum()}"
    )
    
    # Fit view to image
    h, w = slice_data_rgb.shape[:2]
    view.camera.set_range(x=(0, w), y=(0, h))
    view_widget['canvas'].update()


def compose_slice_rgb(viewer_instance, view_name, slice_idx):
    """Return RGB numpy array for a given slice with mask overlays.
    
    Used for 3D slice visualization.
    """
    if viewer_instance.nifti_data is None:
        return None
    
    # Extract slice
    slice_data = _extract_slice(viewer_instance.nifti_data, view_name, slice_idx)
    if slice_data is None:
        return None
    
    # Normalize
    slice_data = slice_data.astype(np.float32)
    if view_name == 'sagittal':
        # Rotate 90° anticlockwise to align orientation
        slice_data = np.rot90(slice_data, k=1)
    
    if slice_data.max() > slice_data.min():
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
    
    slice_rgb = np.stack([slice_data, slice_data, slice_data], axis=2)
    
    # Overlay masks
    if viewer_instance.mask_data:
        composite = slice_rgb.copy()
        for mask, color in zip(viewer_instance.mask_data, viewer_instance.mask_colors):
            # Extract mask slice
            if view_name == 'axial':
                mask_slice = mask[:, :, slice_idx].T
            elif view_name == 'sagittal':
                mask_slice = mask[slice_idx, :, :].T
                mask_slice = np.rot90(mask_slice, k=1)  # Match rotation
            elif view_name == 'coronal':
                mask_slice = mask[:, slice_idx, :].T
                mask_slice = np.fliplr(mask_slice)
            else:
                continue
            
            mask_overlay = mask_slice > 0.5
            if not np.any(mask_overlay):
                continue
            
            alpha = 0.5
            composite[mask_overlay] = (
                (1 - alpha) * composite[mask_overlay] + alpha * color
            )
        slice_rgb = composite
    
    return slice_rgb


def get_slices_with_masks(viewer_instance, view_name):
    """Get list of slice indices that contain masks for a given view.
    
    Uses caching to avoid recomputing.
    """
    if not viewer_instance.mask_data or not viewer_instance.nifti_shape:
        return None
    
    # Check cache key
    cache_key = (view_name, len(viewer_instance.mask_data), 
                 tuple(viewer_instance.nifti_shape))
    if cache_key in viewer_instance._slices_with_masks_cache:
        return viewer_instance._slices_with_masks_cache[cache_key]
    
    slices_with_masks = set()
    
    for mask in viewer_instance.mask_data:
        if view_name == 'axial':
            for z_idx in range(viewer_instance.nifti_shape[2]):
                mask_slice = mask[:, :, z_idx]
                if np.any(mask_slice > 0.5):
                    slices_with_masks.add(z_idx)
        elif view_name == 'sagittal':
            for x_idx in range(viewer_instance.nifti_shape[0]):
                mask_slice = mask[x_idx, :, :]
                if np.any(mask_slice > 0.5):
                    slices_with_masks.add(x_idx)
        elif view_name == 'coronal':
            for y_idx in range(viewer_instance.nifti_shape[1]):
                mask_slice = mask[:, y_idx, :]
                if np.any(mask_slice > 0.5):
                    slices_with_masks.add(y_idx)
    
    result = None
    if slices_with_masks:
        result = sorted(list(slices_with_masks))
    
    # Cache the result
    viewer_instance._slices_with_masks_cache[cache_key] = result
    return result

