"""UI widget creation functions."""

from PyQt5 import QtWidgets, QtCore
from vispy import scene


def create_slice_viewer(view_name):
    """Create a slice viewer widget with canvas, label, and slider.
    
    Args:
        view_name: 'axial', 'sagittal', or 'coronal'
        
    Returns:
        tuple: (slice_container_layout, canvas, slider, label)
    """
    slice_container = QtWidgets.QVBoxLayout()
    
    # Create 2D canvas for slice view
    canvas = scene.SceneCanvas(keys="interactive", bgcolor="black")
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.PanZoomCamera(aspect=1)
    
    slice_container.addWidget(canvas.native)
    
    # Add slider for slice navigation
    label = QtWidgets.QLabel(f"{view_name.capitalize()} Slice: 0")
    label.setStyleSheet("font-weight: bold; color: #0078d4; padding: 3px;")
    label.setAlignment(QtCore.Qt.AlignCenter)
    slice_container.addWidget(label)
    
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(100)
    slider.setValue(50)
    slice_container.addWidget(slider)
    
    return slice_container, canvas, slider, label, view


def create_groupbox(title, parent_layout):
    """Create a styled group box and add it to parent layout.
    
    Args:
        title: Group box title
        parent_layout: Layout to add group box to
        
    Returns:
        tuple: (group_box, layout)
    """
    from ui.styles import GROUPBOX_STYLESHEET
    
    group = QtWidgets.QGroupBox(title)
    group.setStyleSheet(GROUPBOX_STYLESHEET)
    layout = QtWidgets.QVBoxLayout()
    layout.setSpacing(5)
    group.setLayout(layout)
    parent_layout.addWidget(group)
    return group, layout


def create_file_loading_controls(parent_layout, load_folder_cb, load_nifti_cb, load_masks_cb):
    """Create file loading control buttons.
    
    Args:
        parent_layout: Layout to add controls to
        load_folder_cb: Callback for load folder button
        load_nifti_cb: Callback for load NIFTI button
        load_masks_cb: Callback for load masks button
        
    Returns:
        dict: Dictionary with button references
    """
    group, layout = create_groupbox("📁 File Loading", parent_layout)
    
    buttons = {}
    buttons['folder'] = QtWidgets.QPushButton("📦 Load 3D Meshes")
    buttons['folder'].clicked.connect(load_folder_cb)
    layout.addWidget(buttons['folder'])
    
    buttons['nifti'] = QtWidgets.QPushButton("🖼️ Load NIFTI Image")
    buttons['nifti'].clicked.connect(load_nifti_cb)
    layout.addWidget(buttons['nifti'])
    
    buttons['masks'] = QtWidgets.QPushButton("🎭 Load Masks")
    buttons['masks'].clicked.connect(load_masks_cb)
    layout.addWidget(buttons['masks'])
    
    return buttons


def create_medical_imaging_controls(parent_layout, visualize_3d_cb, visualize_slices_cb):
    """Create medical imaging control buttons.
    
    Args:
        parent_layout: Layout to add controls to
        visualize_3d_cb: Callback for visualize 3D button
        visualize_slices_cb: Callback for visualize slices button
        
    Returns:
        dict: Dictionary with button references
    """
    group, layout = create_groupbox("🏥 Medical Imaging", parent_layout)
    
    buttons = {}
    buttons['visualize_3d'] = QtWidgets.QPushButton("🔲 Visualize Masks in 3D")
    buttons['visualize_3d'].clicked.connect(visualize_3d_cb)
    layout.addWidget(buttons['visualize_3d'])
    
    buttons['visualize_slices'] = QtWidgets.QPushButton("📊 Visualize Slices")
    buttons['visualize_slices'].clicked.connect(visualize_slices_cb)
    layout.addWidget(buttons['visualize_slices'])
    
    return buttons


def create_mpr_controls(parent_layout, mark_points_cb, generate_mpr_cb, reset_points_cb):
    """Create MPR control buttons and status label.
    
    Args:
        parent_layout: Layout to add controls to
        mark_points_cb: Callback for mark points button
        generate_mpr_cb: Callback for generate MPR button
        reset_points_cb: Callback for reset points button
        
    Returns:
        dict: Dictionary with button and label references
    """
    group, layout = create_groupbox("📈 Curved MPR", parent_layout)
    
    controls = {}
    controls['mark_points'] = QtWidgets.QPushButton("✏️ Mark Points")
    controls['mark_points'].clicked.connect(mark_points_cb)
    controls['mark_points'].setEnabled(False)
    layout.addWidget(controls['mark_points'])
    
    controls['generate'] = QtWidgets.QPushButton("⚙️ Generate MPR")
    controls['generate'].clicked.connect(generate_mpr_cb)
    controls['generate'].setEnabled(False)
    layout.addWidget(controls['generate'])
    
    controls['reset'] = QtWidgets.QPushButton("🔄 Reset Points")
    controls['reset'].clicked.connect(reset_points_cb)
    controls['reset'].setEnabled(False)
    layout.addWidget(controls['reset'])
    
    controls['status'] = QtWidgets.QLabel("Points: 0")
    controls['status'].setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
    controls['status'].setAlignment(QtCore.Qt.AlignCenter)
    layout.addWidget(controls['status'])
    
    return controls


def create_models_controls(parent_layout, import_gltf_cb, import_gltf_animated_cb, 
                           webengine_available=True):
    """Create 3D models control widgets.
    
    Args:
        parent_layout: Layout to add controls to
        import_gltf_cb: Callback for import GLTF button
        import_gltf_animated_cb: Callback for import animated GLTF button
        webengine_available: Whether WebEngine is available
        
    Returns:
        dict: Dictionary with control references
    """
    group, layout = create_groupbox("🎯 3D Models", parent_layout)
    
    controls = {}
    
    focus_label = QtWidgets.QLabel("Focus on Model:")
    focus_label.setStyleSheet("font-weight: normal; margin-top: 5px;")
    layout.addWidget(focus_label)
    
    controls['dropdown'] = QtWidgets.QComboBox()
    controls['dropdown'].setEnabled(False)
    controls['dropdown'].addItem("None")
    layout.addWidget(controls['dropdown'])
    
    models_list_label = QtWidgets.QLabel("Loaded Models:")
    models_list_label.setStyleSheet("font-weight: normal; margin-top: 5px;")
    layout.addWidget(models_list_label)
    
    controls['list'] = QtWidgets.QListWidget()
    controls['list'].setMaximumHeight(150)
    layout.addWidget(controls['list'])
    
    opacity_label = QtWidgets.QLabel("Opacity:")
    opacity_label.setStyleSheet("font-weight: normal; margin-top: 5px;")
    layout.addWidget(opacity_label)
    
    opacity_layout = QtWidgets.QHBoxLayout()
    controls['opacity_slider'] = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    controls['opacity_slider'].setMinimum(10)
    controls['opacity_slider'].setMaximum(100)
    controls['opacity_slider'].setValue(100)
    controls['opacity_label'] = QtWidgets.QLabel("100%")
    controls['opacity_label'].setMinimumWidth(40)
    opacity_layout.addWidget(controls['opacity_slider'])
    opacity_layout.addWidget(controls['opacity_label'])
    layout.addLayout(opacity_layout)
    
    controls['import_gltf'] = QtWidgets.QPushButton("📂 Import Model (GLTF/GLB)")
    controls['import_gltf'].clicked.connect(import_gltf_cb)
    layout.addWidget(controls['import_gltf'])
    
    controls['import_gltf_animated'] = QtWidgets.QPushButton("🎬 View Animated GLTF/GLB")
    controls['import_gltf_animated'].clicked.connect(import_gltf_animated_cb)
    controls['import_gltf_animated'].setEnabled(webengine_available)
    layout.addWidget(controls['import_gltf_animated'])
    
    # Heart animation button (launches external heart animation script)
    controls['heart_animation'] = QtWidgets.QPushButton("❤️ Heart Animation")
    controls['heart_animation'].setEnabled(True)
    layout.addWidget(controls['heart_animation'])

    return controls


def create_clipping_controls(parent_layout, update_clipping_cb, show_slice_cb):
    """Create clipping planes control widgets.
    
    Args:
        parent_layout: Layout to add controls to
        update_clipping_cb: Callback for clipping updates
        show_slice_cb: Callback for showing slice from clipping (takes plane_name)
        
    Returns:
        dict: Dictionary with control references
    """
    group, layout = create_groupbox("✂️ Clipping Planes", parent_layout)
    
    controls = {}
    
    # Axial plane (Z) - Red
    controls['axial'] = _create_plane_controls('axial', 'Axial (Z)', 'Z', 
                                               '#e74c3c', layout, update_clipping_cb,
                                               lambda: show_slice_cb('axial'))
    
    layout.addSpacing(5)
    
    # Sagittal plane (X) - Blue
    controls['sagittal'] = _create_plane_controls('sagittal', 'Sagittal (X)', 'X',
                                                 '#3498db', layout, update_clipping_cb,
                                                 lambda: show_slice_cb('sagittal'))
    
    layout.addSpacing(5)
    
    # Coronal plane (Y) - Green
    controls['coronal'] = _create_plane_controls('coronal', 'Coronal (Y)', 'Y',
                                                 '#2ecc71', layout, update_clipping_cb,
                                                 lambda: show_slice_cb('coronal'))
    
    return controls


def _create_plane_controls(plane_name, label_text, axis_label, color, 
                           parent_layout, update_cb, show_slice_cb):
    """Helper to create controls for a single clipping plane."""
    from PyQt5 import QtWidgets, QtCore
    
    controls = {}
    
    controls['checkbox'] = QtWidgets.QCheckBox(label_text)
    controls['checkbox'].setChecked(False)
    controls['checkbox'].stateChanged.connect(update_cb)
    parent_layout.addWidget(controls['checkbox'])
    
    controls['label'] = QtWidgets.QLabel(f"{axis_label}: 50%")
    controls['label'].setStyleSheet(f"color: {color}; font-weight: bold;")
    parent_layout.addWidget(controls['label'])
    
    controls['slider'] = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    controls['slider'].setMinimum(0)
    controls['slider'].setMaximum(100)
    controls['slider'].setValue(50)
    controls['slider'].valueChanged.connect(update_cb)
    controls['slider'].valueChanged.connect(
        lambda val: controls['label'].setText(f"{axis_label}: {val}%")
    )
    parent_layout.addWidget(controls['slider'])
    
    controls['show_button'] = QtWidgets.QPushButton("📍 Show on Slices")
    controls['show_button'].setMaximumHeight(25)
    controls['show_button'].clicked.connect(show_slice_cb)
    parent_layout.addWidget(controls['show_button'])
    
    return controls


def create_view_controls(parent_layout, update_scene_cb, reset_camera_cb, clear_views_cb):
    """Create view control buttons.
    
    Args:
        parent_layout: Layout to add controls to
        update_scene_cb: Callback for update scene button
        reset_camera_cb: Callback for reset camera button
        clear_views_cb: Callback for clear all button
        
    Returns:
        dict: Dictionary with button references
    """
    group, layout = create_groupbox("🎮 View Controls", parent_layout)
    
    buttons = {}
    buttons['confirm'] = QtWidgets.QPushButton("✅ Apply Changes")
    buttons['confirm'].clicked.connect(update_scene_cb)
    layout.addWidget(buttons['confirm'])
    
    buttons['reset_camera'] = QtWidgets.QPushButton("🔄 Reset Camera")
    buttons['reset_camera'].clicked.connect(reset_camera_cb)
    layout.addWidget(buttons['reset_camera'])
    
    buttons['clear'] = QtWidgets.QPushButton("🗑️ Clear All")
    buttons['clear'].clicked.connect(clear_views_cb)
    buttons['clear'].setStyleSheet("background-color: #e74c3c; color: white;")
    layout.addWidget(buttons['clear'])
    
    return buttons

