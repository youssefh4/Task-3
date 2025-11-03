"""
Interactive GUI for Curved MPR visualization
Load NIfTI scans, draw paths, generate curved slices

Combined application including:
- Core MPR functions (spline fitting, curve extraction)
- Interactive GUI for visualization and path marking
"""

import sys
import os
import numpy as np
import nibabel as nib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider,
                             QFileDialog, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.ndimage import map_coordinates
from scipy.interpolate import splprep, splev

# Try importing additional medical imaging libraries
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    from skimage import io as skio
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ============================================================================
# Core MPR Functions
# ============================================================================

def fit_spline_curve(points_3d, smoothing=0):
    """
    Fit a B-spline curve through 3D points

    Args:
        points_3d: Nx3 array of 3D coordinates
        smoothing: Smoothing factor (0 = interpolate exactly)

    Returns:
        tck: Spline parameters
        u: Parameter values
    """
    points = np.asarray(points_3d)

    # Determine spline degree (max 3, min 1)
    k = min(3, len(points) - 1)
    k = max(1, k)

    # Fit parametric spline
    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]],
                     s=smoothing, k=k)

    return tck, u


def sample_spline(tck, n_points=500):
    """
    Sample points along fitted spline

    Args:
        tck: Spline parameters from fit_spline_curve
        n_points: Number of points to sample

    Returns:
        Nx3 array of sampled 3D coordinates
    """
    u_samples = np.linspace(0, 1, n_points)
    x, y, z = splev(u_samples, tck)

    return np.column_stack([x, y, z])


def compute_perpendicular_vectors(curve_points):
    """
    Compute perpendicular vectors at each point along curve

    Args:
        curve_points: Nx3 array of points along curve

    Returns:
        Nx3 array of unit perpendicular vectors
    """
    # Calculate tangent via finite differences
    tangent_vectors = np.gradient(curve_points, axis=0)

    # Normalize tangents
    tangent_norms = np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
    tangent_norms = np.where(tangent_norms < 1e-10, 1.0, tangent_norms)
    tangents = tangent_vectors / tangent_norms

    # Compute perpendicular directions
    perpendicular = np.zeros_like(tangents)
    reference = np.array([0, 0, 1.0])  # Default up vector

    for idx in range(len(tangents)):
        # Cross product for perpendicular
        perp = np.cross(tangents[idx], reference)
        perp_norm = np.linalg.norm(perp)

        # Handle parallel case
        if perp_norm < 1e-6:
            reference_alt = np.array([1.0, 0, 0])
            perp = np.cross(tangents[idx], reference_alt)
            perp_norm = np.linalg.norm(perp)

        perpendicular[idx] = perp / (perp_norm + 1e-10)

    return perpendicular


def extract_curved_plane(volume, curve_points, perpendicular_vectors,
                         width=80, order=1):
    """
    Extract 2D slice perpendicular to curve through 3D volume

    Args:
        volume: 3D numpy array
        curve_points: Nx3 array of centerline points
        perpendicular_vectors: Nx3 array of perpendicular directions
        width: Width of extracted slice (pixels)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        2D array: width x len(curve_points) slice
    """
    n_along = len(curve_points)
    result = np.zeros((width, n_along))

    # Offsets from centerline
    offsets = np.linspace(-width / 2, width / 2, width)

    for i, center_pt in enumerate(curve_points):
        perpendicular = perpendicular_vectors[i]

        for j, offset in enumerate(offsets):
            # Calculate sample position
            sample_pos = center_pt + perpendicular * offset

            # Check bounds
            in_bounds = (
                    0 <= sample_pos[0] < volume.shape[0] and
                    0 <= sample_pos[1] < volume.shape[1] and
                    0 <= sample_pos[2] < volume.shape[2]
            )

            if in_bounds:
                # Interpolate intensity
                coords = [[sample_pos[0]], [sample_pos[1]], [sample_pos[2]]]
                value = map_coordinates(volume, coords, order=order,
                                        mode='constant', cval=0.0)
                result[j, i] = value[0]

    return result


def create_curved_mpr(volume_data, path_points, samples=500, width=80):
    """
    Complete pipeline: fit curve and extract MPR slice

    Args:
        volume_data: 3D numpy array
        path_points: List of [x, y, z] coordinates defining path
        samples: Number of samples along curve
        width: Width of perpendicular slice

    Returns:
        2D numpy array of curved MPR, or None on failure
    """
    if len(path_points) < 3:
        print("Error: Need minimum 3 points")
        return None

    try:
        points_array = np.array(path_points)

        # Fit smooth curve
        tck, _ = fit_spline_curve(points_array)

        # Sample curve densely
        sampled_curve = sample_spline(tck, samples)

        # Get perpendicular directions
        perp_vectors = compute_perpendicular_vectors(sampled_curve)

        # Extract slice
        mpr_slice = extract_curved_plane(volume_data, sampled_curve,
                                         perp_vectors, width)

        return mpr_slice

    except Exception as e:
        print(f"MPR generation failed: {e}")
        return None


# ============================================================================
# GUI Classes
# ============================================================================

class ImageCanvas(FigureCanvasQTAgg):
    """Canvas widget for displaying images"""

    def __init__(self, parent=None, width=6, height=6):
        self.fig = Figure(figsize=(width, height), facecolor='#2b2b2b')
        super().__init__(self.fig)
        self.setParent(parent)

        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#2b2b2b')
        self.fig.tight_layout()


class MPRApplication(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Data storage
        self.volume = None
        self.affine_matrix = None
        self.current_slice_idx = 0
        self.marked_points = []
        self.result_image = None

        # Drawing state
        self.is_marking = False

        self.setup_window()
        self.setup_widgets()

    def setup_window(self):
        """Configure main window"""
        self.setWindowTitle("Curved MPR Tool - CT/MRI Viewer")
        self.setGeometry(100, 100, 1500, 850)

        # Dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
                font-size: 13px;
                padding: 3px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 12px 24px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3d3d3d;
                color: #888888;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 6px;
                background: #3d3d3d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #ffffff;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
        """)

    def setup_widgets(self):
        """Create and layout UI components"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header = QLabel("Curved MPR Generator - Medical Imaging Tool")
        header.setStyleSheet("color: #0078d4; font-size: 18px; font-weight: bold;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Control panel
        controls = QHBoxLayout()

        self.load_btn = QPushButton("Load NIfTI File")
        self.load_btn.clicked.connect(self.load_nifti)
        controls.addWidget(self.load_btn)

        self.mark_btn = QPushButton("Mark Points")
        self.mark_btn.clicked.connect(self.toggle_marking)
        self.mark_btn.setEnabled(False)
        controls.addWidget(self.mark_btn)

        self.generate_btn = QPushButton("Generate MPR")
        self.generate_btn.clicked.connect(self.generate_mpr)
        self.generate_btn.setEnabled(False)
        controls.addWidget(self.generate_btn)

        self.reset_btn = QPushButton("Reset Points")
        self.reset_btn.clicked.connect(self.reset_points)
        self.reset_btn.setEnabled(False)
        controls.addWidget(self.reset_btn)

        main_layout.addLayout(controls)

        # Status info
        self.info_label = QLabel("Status: No data loaded. Click 'Load NIfTI File' to begin.")
        self.info_label.setStyleSheet("color: #ffa500; font-size: 12px;")
        main_layout.addWidget(self.info_label)

        # Slice navigation
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Slice Position:"))

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.update_slice)
        nav_layout.addWidget(self.slice_slider)

        self.slice_info = QLabel("-- / --")
        self.slice_info.setStyleSheet("color: #0078d4; font-weight: bold;")
        nav_layout.addWidget(self.slice_info)

        main_layout.addLayout(nav_layout)

        # Image viewers
        splitter = QSplitter(Qt.Horizontal)

        # Left viewer - CT/MRI with path
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        source_label = QLabel("Source Image - Click to Mark Path")
        source_label.setStyleSheet("color: #00d4ff; font-weight: bold; padding: 8px;")
        source_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(source_label)

        self.source_canvas = ImageCanvas(self, width=7, height=7)
        self.source_canvas.mpl_connect('button_press_event', self.on_canvas_click)
        left_layout.addWidget(self.source_canvas)

        splitter.addWidget(left_widget)

        # Right viewer - MPR result
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        result_label = QLabel("Curved MPR Output")
        result_label.setStyleSheet("color: #00d4ff; font-weight: bold; padding: 8px;")
        result_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(result_label)

        self.result_canvas = ImageCanvas(self, width=7, height=7)
        right_layout.addWidget(self.result_canvas)

        splitter.addWidget(right_widget)

        main_layout.addWidget(splitter)

        # Initialize displays
        self.show_placeholder(self.source_canvas, "Load a NIfTI file to begin")
        self.show_placeholder(self.result_canvas, "MPR will appear here")

    def load_nifti(self):
        """Load medical image file - supports multiple formats"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Medical Image File", "",
            "All Supported (*.nii *.nii.gz *.gz *.dcm *.ima *.npy *.npz *.tif *.tiff *.mha *.mhd);;NIfTI (*.nii *.nii.gz *.gz);;NumPy (*.npy *.npz);;DICOM (*.dcm);;TIFF (*.tif *.tiff);;All Files (*.*)"
        )

        if not filepath:
            return

        try:
            self.volume = None
            self.affine_matrix = None

            # Check if file exists, if not try adding extensions
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                print("Trying to fix filename...")

                # Try common extensions
                possible_paths = [
                    filepath + 'z',  # .g -> .gz
                    filepath + '.gz',
                    filepath + '.nii',
                    filepath + '.nii.gz',
                    filepath.replace('.g', '.gz'),
                    filepath.replace('.g', '.nii.gz')
                ]

                for try_path in possible_paths:
                    if os.path.exists(try_path):
                        print(f"[OK] Found file: {try_path}")
                        filepath = try_path
                        break

                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Cannot find file: {filepath}")

            # Get file extension
            _, ext = os.path.splitext(filepath.lower())
            basename = os.path.basename(filepath)

            print(f"Attempting to load: {filepath}")
            print(f"Extension detected: '{ext}'")

            # Method 1: Try NumPy files first (.npy, .npz)
            if ext in ['.npy', '.npz']:
                print("Loading as NumPy array...")
                if ext == '.npy':
                    self.volume = np.load(filepath)
                else:  # .npz
                    data = np.load(filepath)
                    # Try to get the first array
                    key = list(data.keys())[0]
                    self.volume = data[key]
                self.affine_matrix = np.eye(4)
                print(f"[OK] Loaded NumPy array, shape: {self.volume.shape}")

            # Method 2: Try NIfTI files (.nii, .gz)
            elif ext in ['.nii', '.gz'] or '.nii' in filepath.lower():
                print("Loading as NIfTI...")
                nifti_img = nib.load(filepath)
                self.volume = np.asarray(nifti_img.get_fdata())
                self.affine_matrix = nifti_img.affine
                print(f"[OK] Loaded NIfTI, shape: {self.volume.shape}")

            # Method 3: Try DICOM
            elif ext == '.dcm' and PYDICOM_AVAILABLE:
                print("Loading as DICOM...")
                dcm = pydicom.dcmread(filepath)
                self.volume = dcm.pixel_array
                if len(self.volume.shape) == 2:
                    self.volume = self.volume[..., np.newaxis]
                self.affine_matrix = np.eye(4)
                print(f"[OK] Loaded DICOM, shape: {self.volume.shape}")

            # Method 4: Try TIFF stack
            elif ext in ['.tif', '.tiff'] and SKIMAGE_AVAILABLE:
                print("Loading as TIFF...")
                self.volume = skio.imread(filepath)
                self.affine_matrix = np.eye(4)
                print(f"[OK] Loaded TIFF, shape: {self.volume.shape}")

            # Method 5: Try generic image formats
            elif SKIMAGE_AVAILABLE:
                print("Attempting generic image load...")
                self.volume = skio.imread(filepath)
                if len(self.volume.shape) == 2:
                    self.volume = self.volume[..., np.newaxis]
                self.affine_matrix = np.eye(4)
                print(f"[OK] Loaded image, shape: {self.volume.shape}")

            # Method 6: Last resort - try nibabel on any file
            else:
                print("Trying nibabel as last resort...")
                nifti_img = nib.load(filepath)
                self.volume = np.asarray(nifti_img.get_fdata())
                self.affine_matrix = nifti_img.affine
                print(f"[OK] Loaded with nibabel, shape: {self.volume.shape}")

            # Check if we successfully loaded data
            if self.volume is None:
                raise ValueError("Could not load file with any available method")

            # Ensure we have 3D volume
            if len(self.volume.shape) == 2:
                # Convert 2D to 3D
                self.volume = self.volume[..., np.newaxis]
                print(f"Converted 2D to 3D: {self.volume.shape}")
            elif len(self.volume.shape) == 4:
                # Take first volume if 4D
                self.volume = self.volume[:, :, :, 0]
                QMessageBox.information(self, "4D Volume",
                                      "4D volume detected. Using first time point.")
                print(f"Extracted first volume from 4D: {self.volume.shape}")
            elif len(self.volume.shape) != 3:
                raise ValueError(f"Cannot handle shape {self.volume.shape}. Need 3D volume.")

            # Handle different data types and normalize
            if self.volume.dtype == np.bool_:
                self.volume = self.volume.astype(np.float32)
            elif self.volume.dtype not in [np.float64, np.float32]:
                # Normalize integer types
                vmin, vmax = self.volume.min(), self.volume.max()
                if vmax > vmin:
                    self.volume = (self.volume.astype(np.float64) - vmin) / (vmax - vmin)
                else:
                    self.volume = self.volume.astype(np.float64)

            # Set default affine if not set
            if self.affine_matrix is None:
                self.affine_matrix = np.eye(4)

            # Setup slice navigation
            max_slice = self.volume.shape[2] - 1
            self.current_slice_idx = max_slice // 2

            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(max_slice)
            self.slice_slider.setValue(self.current_slice_idx)
            self.slice_slider.setEnabled(True)

            # Enable controls
            self.mark_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)

            # Update display
            self.display_current_slice()

            self.info_label.setText(
                f"Loaded: {os.path.basename(filepath)} | "
                f"Shape: {self.volume.shape} | "
                f"Ready to mark points"
            )
            self.info_label.setStyleSheet("color: #00ff00; font-size: 12px;")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")

    def toggle_marking(self):
        """Toggle point marking mode"""
        self.is_marking = not self.is_marking

        if self.is_marking:
            self.mark_btn.setText("Stop Marking")
            self.mark_btn.setStyleSheet("""
                QPushButton {
                    background-color: #d83b01;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 12px 24px;
                    font-size: 13px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #c13100;
                }
            """)
            self.info_label.setText("Marking ON - Click on image to place points")
            self.info_label.setStyleSheet("color: #ff6b6b; font-size: 12px;")
        else:
            self.mark_btn.setText("Mark Points")
            self.mark_btn.setStyleSheet("")
            count = len(self.marked_points)
            self.info_label.setText(f"Points marked: {count}")
            self.info_label.setStyleSheet("color: #0078d4; font-size: 12px;")

    def on_canvas_click(self, event):
        """Handle mouse click on source canvas"""
        if not self.is_marking or self.volume is None:
            return

        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        # Store 3D coordinate
        x, y = int(event.xdata), int(event.ydata)
        z = self.current_slice_idx

        # Validate bounds
        if (0 <= x < self.volume.shape[0] and
            0 <= y < self.volume.shape[1]):

            self.marked_points.append([x, y, z])
            print(f"Point {len(self.marked_points)}: ({x}, {y}, {z})")

            # Refresh display
            self.display_current_slice()

            # Enable generation if enough points
            if len(self.marked_points) >= 3:
                self.generate_btn.setEnabled(True)
                self.info_label.setText(
                    f"[OK] {len(self.marked_points)} points marked - Ready to generate!"
                )
                self.info_label.setStyleSheet("color: #00ff00; font-size: 12px;")
            else:
                self.info_label.setText(
                    f"Points marked: {len(self.marked_points)} (need 3 minimum)"
                )

    def update_slice(self, value):
        """Update displayed slice when slider moves"""
        self.current_slice_idx = value
        self.display_current_slice()

    def display_current_slice(self):
        """Render current slice with marked points"""
        if self.volume is None:
            return

        # Extract slice
        slice_data = self.volume[:, :, self.current_slice_idx].T

        # Clear and plot
        self.source_canvas.axes.clear()
        self.source_canvas.axes.set_facecolor('#2b2b2b')

        self.source_canvas.axes.imshow(slice_data, cmap='gray',
                                        aspect='equal', interpolation='bilinear')

        # Overlay marked points on current slice
        points_on_slice = [(p[0], p[1]) for p in self.marked_points
                          if p[2] == self.current_slice_idx]

        if points_on_slice:
            pts = np.array(points_on_slice)
            self.source_canvas.axes.scatter(pts[:, 0], pts[:, 1],
                                           c='lime', s=120, marker='x',
                                           linewidths=3, zorder=10)

        # Draw all points with connections
        if len(self.marked_points) > 0:
            all_pts = np.array(self.marked_points)

            # Project all points onto current view
            self.source_canvas.axes.scatter(all_pts[:, 0], all_pts[:, 1],
                                           c='yellow', s=80, alpha=0.6,
                                           edgecolors='red', linewidths=1.5,
                                           zorder=5)

            # Connect with line
            if len(self.marked_points) > 1:
                self.source_canvas.axes.plot(all_pts[:, 0], all_pts[:, 1],
                                            'cyan', linewidth=2, alpha=0.5)

            # Label points
            for i, pt in enumerate(self.marked_points):
                self.source_canvas.axes.text(pt[0], pt[1], str(i+1),
                                            color='white', fontsize=9,
                                            fontweight='bold', ha='center',
                                            va='center', zorder=11)

        self.source_canvas.axes.set_title(
            f"Slice {self.current_slice_idx} / {self.volume.shape[2]-1}",
            color='white', fontsize=11
        )
        self.source_canvas.axes.axis('off')

        self.source_canvas.fig.tight_layout()
        self.source_canvas.draw()

        self.slice_info.setText(f"{self.current_slice_idx} / {self.volume.shape[2]-1}")

    def generate_mpr(self):
        """Generate curved MPR from marked path"""
        if len(self.marked_points) < 3:
            QMessageBox.warning(self, "Insufficient Points",
                              "Please mark at least 3 points on the image.")
            return

        self.info_label.setText("Generating Curved MPR...")
        self.info_label.setStyleSheet("color: #ffa500; font-size: 12px;")
        QApplication.processEvents()

        # Call core MPR function
        self.result_image = create_curved_mpr(
            self.volume,
            self.marked_points,
            samples=400,
            width=100
        )

        if self.result_image is not None:
            self.display_mpr_result()
            self.info_label.setText(
                f"[OK] Curved MPR generated! Shape: {self.result_image.shape}"
            )
            self.info_label.setStyleSheet("color: #00ff00; font-size: 12px;")
        else:
            QMessageBox.critical(self, "Generation Failed",
                               "Could not generate MPR. Check console for errors.")
            self.info_label.setText("ERROR: MPR generation failed")
            self.info_label.setStyleSheet("color: #ff0000; font-size: 12px;")

    def display_mpr_result(self):
        """Display generated MPR image"""
        if self.result_image is None:
            return

        self.result_canvas.axes.clear()
        self.result_canvas.axes.set_facecolor('#2b2b2b')

        im = self.result_canvas.axes.imshow(self.result_image, cmap='gray',
                                           aspect='auto', interpolation='bilinear')

        self.result_canvas.axes.set_title('Curved MPR Result',
                                         color='white', fontsize=12,
                                         fontweight='bold')
        self.result_canvas.axes.set_xlabel('Distance Along Path',
                                          color='white', fontsize=10)
        self.result_canvas.axes.set_ylabel('Perpendicular Width',
                                          color='white', fontsize=10)
        self.result_canvas.axes.tick_params(colors='white', labelsize=9)

        # Add colorbar
        cbar = self.result_canvas.fig.colorbar(im, ax=self.result_canvas.axes)
        cbar.ax.tick_params(colors='white', labelsize=8)

        self.result_canvas.fig.tight_layout()
        self.result_canvas.draw()

    def reset_points(self):
        """Clear all marked points"""
        self.marked_points.clear()
        self.result_image = None

        self.generate_btn.setEnabled(False)

        if self.volume is not None:
            self.display_current_slice()

        self.show_placeholder(self.result_canvas, "MPR will appear here")

        self.info_label.setText("All points cleared")
        self.info_label.setStyleSheet("color: #ffa500; font-size: 12px;")

        if self.is_marking:
            self.toggle_marking()

    def show_placeholder(self, canvas, text):
        """Show placeholder text on canvas"""
        canvas.axes.clear()
        canvas.axes.set_facecolor('#2b2b2b')
        canvas.axes.text(0.5, 0.5, text, ha='center', va='center',
                        color='#666666', fontsize=13,
                        transform=canvas.axes.transAxes)
        canvas.axes.set_xticks([])
        canvas.axes.set_yticks([])
        for spine in canvas.axes.spines.values():
            spine.set_color('#444444')
        canvas.fig.tight_layout()
        canvas.draw()


def main():
    """Launch application"""
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        window = MPRApplication()
        window.show()

        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        error_msg = f"Error launching application:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        # Try to show error in a message box if possible
        try:
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Application Error")
            msg.setText("Failed to launch application")
            msg.setDetailedText(error_msg)
            msg.exec_()
        except:
            input("Press Enter to exit...")  # Keep console open on Windows


if __name__ == '__main__':
    main()