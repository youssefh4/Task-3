"""MPR (Multi-Planar Reconstruction) functionality."""

import numpy as np
from PyQt5 import QtWidgets, QtCore

try:
    from cmpr_gui_v2 import create_curved_mpr
except ImportError:
    print("Warning: cmpr_gui_v2 not available. MPR functionality may be limited.")


def setup_axial_click_handler(viewer_instance):
    """Connect click handler to axial view."""
    view_widget = viewer_instance.slice_widgets['axial']
    view = view_widget['view']
    try:
        view.events.mouse_press.disconnect()
    except:
        pass
    
    @view.events.mouse_press.connect
    def on_axial_mouse_press(event):
        handle_axial_click(viewer_instance, event)


def handle_axial_click(viewer_instance, event):
    """Handle mouse click on axial slice canvas to mark points."""
    if not viewer_instance.is_marking_points or viewer_instance.nifti_data is None:
        return
    
    current_z = viewer_instance.slice_sliders['axial'].value()
    view_widget = viewer_instance.slice_widgets['axial']
    view = view_widget['view']
    cam_rect = view.camera.rect
    if cam_rect is None:
        return
    
    # Get click position and transform to scene coordinates
    click_pos = event.pos
    canvas = view.canvas
    
    # Transform to scene coordinates
    scene_x = cam_rect.left + (click_pos[0] / canvas.size[0]) * cam_rect.width
    scene_y = (cam_rect.bottom + 
               ((canvas.size[1] - click_pos[1]) / canvas.size[1]) * cam_rect.height)
    
    # Map to image coordinates
    h, w = viewer_instance.nifti_shape[1], viewer_instance.nifti_shape[0]
    
    rel_x = (scene_x - cam_rect.left) / cam_rect.width if cam_rect.width > 0 else 0
    rel_y = (scene_y - cam_rect.bottom) / cam_rect.height if cam_rect.height > 0 else 0
    rel_x = max(0, min(1, rel_x))
    rel_y = max(0, min(1, rel_y))
    
    # Get pixel coordinates
    img_x = int(rel_x * w)
    img_y = int(rel_y * h)
    img_x = max(0, min(img_x, w - 1))
    img_y = max(0, min(img_y, h - 1))
    
    # Convert to NIFTI coordinates
    nifti_x = img_x
    nifti_y = img_y
    nifti_z = current_z
    
    # Validate and add point
    if (0 <= nifti_x < viewer_instance.nifti_shape[0] and
        0 <= nifti_y < viewer_instance.nifti_shape[1] and
        0 <= nifti_z < viewer_instance.nifti_shape[2]):
        viewer_instance.marked_points.append([nifti_x, nifti_y, nifti_z])
        print(f"Marked point {len(viewer_instance.marked_points)}: "
              f"({nifti_x}, {nifti_y}, {nifti_z})")
        from slices.viewer import update_slice_display
        update_slice_display(viewer_instance, 'axial', current_z)
        update_mpr_status(viewer_instance)
        if len(viewer_instance.marked_points) >= 3:
            viewer_instance.mpr_controls['generate'].setEnabled(True)


def toggle_marking_points(viewer_instance):
    """Toggle point marking mode."""
    viewer_instance.is_marking_points = not viewer_instance.is_marking_points
    if viewer_instance.is_marking_points:
        viewer_instance.mpr_controls['mark_points'].setText("Stop Marking")
        viewer_instance.mpr_controls['mark_points'].setStyleSheet(
            "background-color: #d83b01; color: white;"
        )
        viewer_instance.mpr_controls['status'].setText(
            "Marking ON - Click on axial slice to place points"
        )
        viewer_instance.mpr_controls['status'].setStyleSheet("color: #ff6b6b;")
        # Reconnect click handler
        if 'axial' in viewer_instance.slice_widgets:
            setup_axial_click_handler(viewer_instance)
            # Refresh display
            current_z = viewer_instance.slice_sliders['axial'].value()
            from slices.viewer import update_slice_display
            update_slice_display(viewer_instance, 'axial', current_z)
    else:
        viewer_instance.mpr_controls['mark_points'].setText("✏️ Mark Points")
        viewer_instance.mpr_controls['mark_points'].setStyleSheet("")
        update_mpr_status(viewer_instance)


def update_mpr_status(viewer_instance):
    """Update MPR status label."""
    count = len(viewer_instance.marked_points)
    if count >= 3:
        viewer_instance.mpr_controls['status'].setText(
            f"Points: {count} (Ready to generate)"
        )
        viewer_instance.mpr_controls['status'].setStyleSheet("color: #00ff00;")
    else:
        viewer_instance.mpr_controls['status'].setText(
            f"Points: {count} (need 3 minimum)"
        )
        viewer_instance.mpr_controls['status'].setStyleSheet("color: #ffa500;")


def generate_mpr(viewer_instance):
    """Generate curved MPR from marked points."""
    if len(viewer_instance.marked_points) < 3:
        QtWidgets.QMessageBox.warning(
            viewer_instance, "Insufficient Points",
            "Please mark at least 3 points on the axial slice."
        )
        return
    
    if viewer_instance.nifti_data is None:
        QtWidgets.QMessageBox.warning(
            viewer_instance, "No NIFTI Loaded",
            "Please load a NIFTI file first."
        )
        return
    
    try:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        viewer_instance.mpr_controls['status'].setText("Generating Curved MPR...")
        viewer_instance.mpr_controls['status'].setStyleSheet("color: #ffa500;")
        QtWidgets.QApplication.processEvents()
        
        # Use imported function from cmpr_gui_v2
        try:
            viewer_instance.mpr_result = create_curved_mpr(
                viewer_instance.nifti_data,
                viewer_instance.marked_points,
                samples=400,
                width=100
            )
        except NameError:
            QtWidgets.QMessageBox.critical(
                viewer_instance, "Import Error",
                "create_curved_mpr function not available. "
                "Please ensure cmpr_gui_v2 is accessible."
            )
            return
        
        if viewer_instance.mpr_result is not None:
            display_mpr_result(viewer_instance)
            viewer_instance.mpr_controls['status'].setText(
                f"MPR generated! Shape: {viewer_instance.mpr_result.shape}"
            )
            viewer_instance.mpr_controls['status'].setStyleSheet("color: #00ff00;")
        else:
            QtWidgets.QMessageBox.critical(
                viewer_instance, "Generation Failed",
                "Could not generate MPR. Check console for errors."
            )
            viewer_instance.mpr_controls['status'].setText("ERROR: MPR generation failed")
            viewer_instance.mpr_controls['status'].setStyleSheet("color: #ff0000;")
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            viewer_instance, "Error", f"Failed to generate MPR:\n{str(e)}"
        )
        import traceback
        traceback.print_exc()
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


def display_mpr_result(viewer_instance):
    """Display generated MPR image in a new window."""
    if viewer_instance.mpr_result is None:
        return
    
    # Close previous window if exists
    if viewer_instance.mpr_window is not None:
        try:
            viewer_instance.mpr_window.close()
        except:
            pass
    
    # Create new window
    viewer_instance.mpr_window = QtWidgets.QWidget()
    viewer_instance.mpr_window.setWindowTitle("Curved MPR Result")
    viewer_instance.mpr_window.setGeometry(100, 100, 800, 600)
    layout = QtWidgets.QVBoxLayout(viewer_instance.mpr_window)
    layout.setContentsMargins(10, 10, 10, 10)
    
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    
    fig = Figure(figsize=(8, 6), facecolor='black')
    canvas = FigureCanvasQTAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    
    img = viewer_instance.mpr_result.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    
    im = ax.imshow(img, cmap='gray', aspect='auto', interpolation='bilinear')
    ax.set_title('Curved MPR Result', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Distance Along Path', color='white', fontsize=10)
    ax.set_ylabel('Perpendicular Width', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=9)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors='white', labelsize=8)
    fig.tight_layout()
    layout.addWidget(canvas)
    
    save_btn = QtWidgets.QPushButton("Save MPR Image")
    save_btn.clicked.connect(lambda: save_mpr_image(viewer_instance))
    layout.addWidget(save_btn)
    
    # Show window
    viewer_instance.mpr_window.setWindowModality(QtCore.Qt.NonModal)
    viewer_instance.mpr_window.show()
    viewer_instance.mpr_window.raise_()
    viewer_instance.mpr_window.activateWindow()


def save_mpr_image(viewer_instance):
    """Save MPR result to file."""
    if viewer_instance.mpr_result is None:
        QtWidgets.QMessageBox.warning(viewer_instance, "No MPR", "No MPR result to save.")
        return
    
    file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        viewer_instance, "Save MPR Image", "", "NumPy files (*.npy);;All Files (*.*)"
    )
    if file_path:
        try:
            np.save(file_path, viewer_instance.mpr_result)
            QtWidgets.QMessageBox.information(
                viewer_instance, "Saved", f"MPR saved to {file_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                viewer_instance, "Error", f"Failed to save:\n{str(e)}"
            )


def reset_mpr_points(viewer_instance):
    """Clear all marked points."""
    viewer_instance.marked_points.clear()
    viewer_instance.mpr_result = None
    viewer_instance.mpr_controls['generate'].setEnabled(False)
    if viewer_instance.nifti_data is not None:
        current_z = viewer_instance.slice_sliders['axial'].value()
        from slices.viewer import update_slice_display
        update_slice_display(viewer_instance, 'axial', current_z)
    update_mpr_status(viewer_instance)
    if viewer_instance.is_marking_points:
        toggle_marking_points(viewer_instance)

