"""MPR window with axial view and controls."""

import numpy as np
from PyQt5 import QtWidgets, QtCore
from vispy import scene
from ui.widgets import create_slice_viewer
from slices.viewer import update_slice_display, setup_slice_sliders, _extract_slice, _overlay_mpr_points

try:
    from cmpr_gui_v2 import create_curved_mpr
except ImportError:
    print("Warning: cmpr_gui_v2 not available. MPR functionality may be limited.")
    create_curved_mpr = None


class MPRWindow(QtWidgets.QWidget):
    """Window for Curved MPR point marking and generation."""
    
    def __init__(self, parent_viewer):
        super().__init__()
        self.parent_viewer = parent_viewer
        self.is_marking_points = False
        self.marked_points = []
        self.mpr_result = None
        self._init_ui()
        self._connect_signals()
        
        # Setup click handler
        self._setup_click_handler()
    
    def _init_ui(self):
        """Initialize the MPR window UI."""
        self.setWindowTitle("Curved MPR - Point Marking")
        self.setGeometry(100, 100, 900, 700)
        
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side: Axial view
        left_layout = QtWidgets.QVBoxLayout()
        
        # Create axial slice viewer
        slice_container, canvas, slider, label, view = create_slice_viewer('axial')
        self.axial_canvas = canvas
        self.axial_view = view
        self.axial_slider = slider
        self.axial_label = label
        
        left_layout.addWidget(canvas.native, 1)
        left_layout.addWidget(label, 0)
        left_layout.addWidget(slider, 0)
        
        # Right side: Controls
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setSpacing(10)
        
        # Title
        title = QtWidgets.QLabel("Curved MPR Controls")
        title.setStyleSheet("font-weight: bold; font-size: 12pt; margin-bottom: 10px;")
        controls_layout.addWidget(title)
        
        # Mark Points button
        self.mark_button = QtWidgets.QPushButton("Mark Points")
        self.mark_button.clicked.connect(self.toggle_marking)
        self.mark_button.setMinimumHeight(30)
        controls_layout.addWidget(self.mark_button)
        
        # Generate MPR button
        self.generate_button = QtWidgets.QPushButton("Generate MPR")
        self.generate_button.clicked.connect(self.generate)
        self.generate_button.setEnabled(False)
        self.generate_button.setMinimumHeight(30)
        controls_layout.addWidget(self.generate_button)
        
        # Reset Points button
        self.reset_button = QtWidgets.QPushButton("Reset Points")
        self.reset_button.clicked.connect(self.reset_points)
        self.reset_button.setEnabled(False)
        self.reset_button.setMinimumHeight(30)
        controls_layout.addWidget(self.reset_button)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Points: 0")
        self.status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setMinimumHeight(30)
        controls_layout.addWidget(self.status_label)
        
        controls_layout.addStretch()
        
        # Add layouts to main layout
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(controls_layout, 1)
        
        self.setLayout(main_layout)
        
        # Load initial slice if data is available
        if self.parent_viewer.nifti_data is not None:
            self._setup_slider()
            self._update_display()
    
    def _setup_slider(self):
        """Setup slider ranges based on NIFTI dimensions."""
        if self.parent_viewer.nifti_data is None:
            return
        
        axial_max = max(0, self.parent_viewer.nifti_shape[2] - 1)
        self.axial_slider.setMinimum(0)
        self.axial_slider.setMaximum(axial_max)
        self.axial_slider.setValue(self.parent_viewer.nifti_shape[2] // 2)
    
    def _connect_signals(self):
        """Connect slider to update function."""
        self.axial_slider.valueChanged.connect(self._on_slider_changed)
    
    def _on_slider_changed(self, value):
        """Handle slider value change."""
        self._update_display()
    
    def _update_display(self):
        """Update the displayed slice."""
        if self.parent_viewer.nifti_data is None:
            return
        
        current_z = self.axial_slider.value()
        
        # Update label
        self.axial_label.setText(
            f"Axial: {current_z}/{self.axial_slider.maximum()}"
        )
        
        # Extract slice
        slice_data = _extract_slice(self.parent_viewer.nifti_data, 'axial', current_z)
        
        # Normalize
        slice_data = slice_data.astype(np.float32)
        if slice_data.max() > slice_data.min():
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        
        # Convert to RGB
        slice_data_rgb = np.stack([slice_data, slice_data, slice_data], axis=2)
        
        # Overlay marked points
        if self.marked_points:
            slice_data_rgb = self._overlay_points(current_z, slice_data_rgb)
        
        # Display
        self._display_image(slice_data_rgb)
    
    def _overlay_points(self, slice_idx, slice_data_rgb):
        """Overlay marked points on slice."""
        # Displayed image is nifti_data[:, :, z].T
        # So displayed[row, col] = NIFTI[col, row, z]
        # Points are stored as [x, y, z] in NIFTI coordinates
        for pt in self.marked_points:
            if int(pt[2]) == slice_idx:
                nifti_x = int(pt[0])  # X in NIFTI
                nifti_y = int(pt[1])  # Y in NIFTI
                # Convert to displayed image coordinates
                # displayed[row, col] = NIFTI[col, row]
                # So: displayed row = NIFTI y, displayed col = NIFTI x
                displayed_row = nifti_y  # Row in displayed image = Y in NIFTI
                displayed_col = nifti_x  # Col in displayed image = X in NIFTI
                
                size = 5
                for dy in range(-size, size+1):
                    for dx in range(-size, size+1):
                        if abs(dy) == abs(dx) or dy == 0 or dx == 0:
                            py, px = displayed_row + dy, displayed_col + dx
                            if (0 <= py < slice_data_rgb.shape[0] and
                                0 <= px < slice_data_rgb.shape[1]):
                                slice_data_rgb[py, px] = [1.0, 1.0, 0.0]  # Yellow
        return slice_data_rgb
    
    def _display_image(self, slice_data_rgb):
        """Display slice image in the viewer."""
        # Remove existing image
        for child in list(self.axial_view.scene.children):
            if isinstance(child, scene.visuals.Image):
                child.parent = None
        
        # Create new image
        image = scene.visuals.Image(slice_data_rgb, parent=self.axial_view.scene)
        
        # Fit view to image
        h, w = slice_data_rgb.shape[:2]
        self.axial_view.camera.set_range(x=(0, w), y=(0, h))
        self.axial_canvas.update()
    
    def _setup_click_handler(self):
        """Setup click handler for axial view."""
        try:
            self.axial_view.events.mouse_press.disconnect()
        except:
            pass
        
        @self.axial_view.events.mouse_press.connect
        def on_axial_mouse_press(event):
            self._handle_click(event)
    
    def _handle_click(self, event):
        """Handle mouse click on axial slice."""
        if not self.is_marking_points or self.parent_viewer.nifti_data is None:
            return
        
        current_z = self.axial_slider.value()
        click_pos = event.pos
        
        # Get viewport size
        viewport_size = self.axial_view.size
        if viewport_size is None or viewport_size[0] == 0 or viewport_size[1] == 0:
            return
        
        # Get camera rect
        cam_rect = self.axial_view.camera.rect
        if cam_rect is None:
            return
        
        # Use vispy's proper coordinate transformation
        # Transform viewport coordinates to scene coordinates using the camera
        try:
            # Use the camera's inverse transform to map viewport to scene
            # For PanZoomCamera, transform.imap maps from viewport to scene
            viewport_pos = np.array([click_pos[0], click_pos[1], 0.0])
            scene_pos = self.axial_view.camera.transform.imap(viewport_pos)
            scene_x = float(scene_pos[0])
            scene_y = float(scene_pos[1])
        except Exception as e:
            # Fallback to manual calculation using camera rect
            norm_x = click_pos[0] / viewport_size[0] if viewport_size[0] > 0 else 0
            norm_y = click_pos[1] / viewport_size[1] if viewport_size[1] > 0 else 0
            scene_x = cam_rect.left + norm_x * cam_rect.width
            scene_y = cam_rect.bottom + (1.0 - norm_y) * cam_rect.height
        
        # Get displayed image dimensions
        # Displayed image is nifti_data[:, :, z].T
        # nifti_data[:, :, z] has shape [shape[0], shape[1]] = [x_size, y_size]
        # After .T: displayed image has shape [shape[1], shape[0]] = [y_size, x_size]
        # Camera range is set to x=(0, shape[1]), y=(0, shape[0])
        displayed_width = float(self.parent_viewer.nifti_shape[1])  # Y dimension in NIFTI
        displayed_height = float(self.parent_viewer.nifti_shape[0])  # X dimension in NIFTI
        
        # Convert scene coordinates to displayed image pixel coordinates
        # scene_x is in [cam_rect.left, cam_rect.left + cam_rect.width] = [0, displayed_width]
        # scene_y is in [cam_rect.bottom, cam_rect.bottom + cam_rect.height] = [0, displayed_height]
        # Use round for better sub-pixel accuracy
        displayed_col = int(round(scene_x))  # Column in displayed image
        displayed_row = int(round(scene_y))  # Row in displayed image
        
        # Clamp to displayed image bounds
        displayed_col = max(0, min(displayed_col, int(displayed_width) - 1))
        displayed_row = max(0, min(displayed_row, int(displayed_height) - 1))
        
        # Convert displayed image coordinates to NIFTI coordinates
        # Displayed image is nifti_data[:, :, z].T
        # So displayed[row, col] = NIFTI[col, row, z]
        # Therefore:
        # - displayed_col = X in NIFTI
        # - displayed_row = Y in NIFTI
        nifti_x = displayed_col  # X in NIFTI = column in displayed image
        nifti_y = displayed_row  # Y in NIFTI = row in displayed image
        nifti_z = current_z
        
        # Validate and add point
        if (0 <= nifti_x < self.parent_viewer.nifti_shape[0] and
            0 <= nifti_y < self.parent_viewer.nifti_shape[1] and
            0 <= nifti_z < self.parent_viewer.nifti_shape[2]):
            self.marked_points.append([nifti_x, nifti_y, nifti_z])
            print(f"Marked point {len(self.marked_points)}: ({nifti_x}, {nifti_y}, {nifti_z})")
            self._update_display()
            self._update_status()
            if len(self.marked_points) >= 3:
                self.generate_button.setEnabled(True)
    
    def toggle_marking(self):
        """Toggle point marking mode."""
        self.is_marking_points = not self.is_marking_points
        if self.is_marking_points:
            self.mark_button.setText("Stop Marking")
            self.mark_button.setStyleSheet("background-color: #d83b01; color: white;")
            self.status_label.setText("Marking ON - Click on image to place points")
            self.status_label.setStyleSheet("color: #ff6b6b; padding: 5px; background-color: #fff0f0;")
            self.reset_button.setEnabled(True)
        else:
            self.mark_button.setText("Mark Points")
            self.mark_button.setStyleSheet("")
            self._update_status()
    
    def _update_status(self):
        """Update status label."""
        count = len(self.marked_points)
        if count >= 3:
            self.status_label.setText(f"Points: {count} (Ready to generate)")
            self.status_label.setStyleSheet("color: #00aa00; padding: 5px; background-color: #f0fff0;")
        else:
            self.status_label.setText(f"Points: {count} (need 3 minimum)")
            self.status_label.setStyleSheet("color: #ffa500; padding: 5px; background-color: #fff8f0;")
    
    def generate(self):
        """Generate curved MPR."""
        if len(self.marked_points) < 3:
            QtWidgets.QMessageBox.warning(
                self, "Insufficient Points",
                "Please mark at least 3 points on the axial slice."
            )
            return
        
        if self.parent_viewer.nifti_data is None:
            QtWidgets.QMessageBox.warning(
                self, "No NIFTI Loaded",
                "Please load a NIFTI file first."
            )
            return
        
        if create_curved_mpr is None:
            QtWidgets.QMessageBox.critical(
                self, "Import Error",
                "create_curved_mpr function not available. Please ensure cmpr_gui_v2 is accessible."
            )
            return
        
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.status_label.setText("Generating Curved MPR...")
            self.status_label.setStyleSheet("color: #ffa500; padding: 5px; background-color: #fff8f0;")
            QtWidgets.QApplication.processEvents()
            
            # Generate MPR using the marked points
            self.mpr_result = create_curved_mpr(
                self.parent_viewer.nifti_data,
                self.marked_points,
                samples=400,
                width=100
            )
            
            if self.mpr_result is not None:
                # Store in parent viewer for compatibility
                self.parent_viewer.marked_points = self.marked_points
                self.parent_viewer.mpr_result = self.mpr_result
                
                # Display result
                self._display_mpr_result()
                self.status_label.setText(f"MPR generated! Shape: {self.mpr_result.shape}")
                self.status_label.setStyleSheet("color: #00aa00; padding: 5px; background-color: #f0fff0;")
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Generation Failed",
                    "Could not generate MPR. Check console for errors."
                )
                self.status_label.setText("ERROR: MPR generation failed")
                self.status_label.setStyleSheet("color: #ff0000; padding: 5px; background-color: #fff0f0;")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to generate MPR:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    
    def _display_mpr_result(self):
        """Display generated MPR image in a new window."""
        if self.mpr_result is None:
            return
        
        # Close previous window if exists
        if hasattr(self.parent_viewer, 'mpr_window_result') and self.parent_viewer.mpr_window_result is not None:
            try:
                self.parent_viewer.mpr_window_result.close()
            except:
                pass
        
        # Create new window
        from PyQt5 import QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        
        mpr_window = QtWidgets.QWidget()
        mpr_window.setWindowTitle("Curved MPR Result")
        mpr_window.setGeometry(100, 100, 800, 600)
        layout = QtWidgets.QVBoxLayout(mpr_window)
        layout.setContentsMargins(10, 10, 10, 10)
        
        fig = Figure(figsize=(8, 6), facecolor='black')
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        
        img = self.mpr_result.astype(np.float32)
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
        save_btn.clicked.connect(lambda: self._save_mpr_image())
        layout.addWidget(save_btn)
        
        mpr_window.setWindowModality(QtCore.Qt.NonModal)
        mpr_window.show()
        mpr_window.raise_()
        mpr_window.activateWindow()
        
        # Store reference
        self.parent_viewer.mpr_window_result = mpr_window
    
    def _save_mpr_image(self):
        """Save MPR result to file."""
        if self.mpr_result is None:
            QtWidgets.QMessageBox.warning(self, "No MPR", "No MPR result to save.")
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save MPR Image", "", "NumPy files (*.npy);;All Files (*.*)"
        )
        if file_path:
            try:
                np.save(file_path, self.mpr_result)
                QtWidgets.QMessageBox.information(
                    self, "Saved", f"MPR saved to {file_path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to save:\n{str(e)}"
                )
    
    def reset_points(self):
        """Clear all marked points."""
        self.marked_points.clear()
        self.mpr_result = None
        self.generate_button.setEnabled(False)
        self._update_display()
        self._update_status()
        if self.is_marking_points:
            self.toggle_marking()

