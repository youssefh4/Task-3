# 🏥 3D Medical Image Viewer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15-green?logo=qt)
![VisPy](https://img.shields.io/badge/VisPy-Latest-orange)
![License](https://img.shields.io/badge/License-TBD-lightgrey)

**A comprehensive, interactive 3D medical imaging application built with PyQt5 and VisPy**

*Advanced visualization capabilities for medical images, 3D meshes, and multi-planar reconstruction (MPR) with curved path generation*

</div>

---

## 📑 Table of Contents

- [Screenshots](#-screenshots)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Demo Videos](#-demo-videos)
- [Project Structure](#-project-structure)
- [Architecture](#️-architecture)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 📸 Screenshots

<div align="center">

### Application Interface

![Screenshot 1](https://github.com/user-attachments/assets/648227c2-c1e5-4e98-aec7-a3b645a1a02b)
*Main application interface with 3D visualization*

![Screenshot 2](https://github.com/user-attachments/assets/3ce5025d-f263-4450-8420-7b365a276bd1)
*Multi-planar reconstruction view*

![Screenshot 3](https://github.com/user-attachments/assets/e61d920f-8ed2-4edc-a281-7e678783a076)
*Slice viewer with mask overlay*

![Screenshot 4](https://github.com/user-attachments/assets/8174b6ab-a4ef-459f-aaae-886286ae27ac)
*3D mesh rendering*

![Screenshot 5](https://github.com/user-attachments/assets/186e881e-87a0-4771-961d-3d876e6abd20)
*Curved MPR visualization*

![Screenshot 6](https://github.com/user-attachments/assets/05435ed1-d69f-441a-8741-4a9f42bf5c00)
*Interactive clipping planes*

![Screenshot 7](https://github.com/user-attachments/assets/45341a03-1c98-4bc9-9252-88ac1c71dc7d)
*Advanced 3D scene management*

</div>

## 🎯 Features

### Core Capabilities
- **3D Medical Image Visualization**: Load and visualize NIFTI medical imaging files (.nii, .nii.gz)
- **Multi-Planar Reconstruction (MPR)**: Generate curved MPR slices by marking points on axial slices
- **3D Mesh Rendering**: Support for multiple mesh formats (STL, OBJ, GLB, GLTF)
- **Interactive Clipping Planes**: Create and manipulate clipping planes to explore internal structures
- **Slice Viewers**: Simultaneous display of axial, sagittal, and coronal slices with mask overlays
- **3D Slice Visualization**: Display perpendicular slice images directly in the 3D scene
- **Mask Overlay**: Visualize segmentation masks on 2D slices and in 3D space
- **GLTF/GLB Support**: Import static models via trimesh and animated models via Three.js

### Advanced Features
- **Curved MPR Generation**: Mark points on slices to create curved multi-planar reconstructions
- **Real-time Updates**: Synchronized updates between 2D slice viewers and 3D scene
- **Camera Controls**: Interactive camera manipulation with reset and focus capabilities
- **Scene Management**: Automatic bounds computation and scene organization
- **Settings Persistence**: Application settings saved between sessions

## 📋 Requirements

### Core Dependencies

| Package | Purpose |
|---------|---------|
| **Python** | 3.8+ (recommended: 3.11 or 3.12) |
| **PyQt5** | GUI framework |
| **VisPy** | 3D visualization engine |
| **NumPy** | Numerical computing |
| **Nibabel** | NIFTI file handling |
| **SciPy** | Scientific computing (spline fitting, interpolation) |
| **Matplotlib** | 2D plotting and visualization |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| **Trimesh** | Enhanced mesh file support (GLTF, GLB) |
| **PyVista** | Additional 3D visualization capabilities |
| **PyDICOM** | DICOM file support |
| **scikit-image** | Image processing utilities |
| **PyQtWebEngine** | Animated GLTF model viewer |

## 🚀 Installation

### Using Conda (Recommended)

```bash
# Create and activate a new conda environment
conda create -n medical-viewer python=3.11 -y
conda activate medical-viewer

# Install core dependencies
conda install -c conda-forge pyqt vispy numpy scipy matplotlib nibabel -y

# Install additional dependencies via pip
pip install trimesh pyvista
```

### Using pip

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install PyQt5 vispy numpy scipy matplotlib nibabel trimesh pyvista
```

### Optional Dependencies

```bash
# For DICOM support
pip install pydicom

# For additional image processing
pip install scikit-image

# For animated GLTF models (requires PyQtWebEngine)
pip install PyQtWebEngine
```

## 💻 Usage

### Quick Start

```bash
# Clone the repository (if applicable)
git clone <repository-url>
cd Task-3

# Install dependencies (see Installation section)
# Then run the application
python main.py
```

### Running the Application

Simply run the main application:

```bash
python main.py
```

### Basic Workflow

1. **Load Medical Images**: 
   - Click "Load NIFTI File" to load medical imaging data
   - Navigate through slices using the sliders for axial, sagittal, and coronal views

2. **Load 3D Meshes**:
   - Click "Load Mesh Folder" to load 3D mesh files (STL, OBJ, GLB, GLTF)
   - Meshes are automatically rendered in the 3D scene

3. **Load Segmentation Masks**:
   - Click "Load Masks Folder" to load mask files
   - Masks can be overlaid on 2D slices or visualized in 3D

4. **Create Curved MPR**:
   - Load a NIFTI file
   - Enable "Mark Points" mode
   - Click points on the axial slice viewer to define a path
   - Click "Generate MPR" to create curved multi-planar reconstruction

5. **Use Clipping Planes**:
   - Create clipping planes to explore internal structures
   - Adjust plane positions using the controls
   - View slices from clipping planes in the 2D viewers

6. **View Controls**:
   - Use "Reset Camera" to return to default view
   - Use "Focus on Model" to center the camera on loaded meshes
   - Adjust scene settings for optimal visualization




---

## 🎥 Demo Videos

<div align="center">

### Application Demo

[![Demo Video 1](https://img.shields.io/badge/▶️-Watch%20Demo%201-blue)](https://github.com/user-attachments/assets/a6450d80-92be-421e-abc6-7a7b71f374d6)

*Interactive 3D visualization and MPR generation*

[![Demo Video 2](https://img.shields.io/badge/▶️-Watch%20Demo%202-blue)](https://github.com/user-attachments/assets/47464d10-3f0f-474a-ba0d-a68abbe97540)

*Advanced features and workflow demonstration*

</div>










## 📁 Project Structure

```
Task-3/
├── main.py                    # Main application entry point (STLViewer class)
├── cmpr_gui_v2.py            # Legacy GUI implementation (reference)
│
├── ui/                       # User Interface Components
│   ├── __init__.py
│   ├── styles.py             # Application-wide styling and themes
│   └── widgets.py            # Widget creation functions (buttons, sliders, etc.)
│
├── loaders/                  # File Loading Module
│   ├── __init__.py
│   └── file_handlers.py      # Handlers for:
│                             #   • 3D mesh files (STL, OBJ, GLB, GLTF)
│                             #   • NIFTI medical images
│                             #   • Segmentation masks
│
├── visualization/            # 3D Visualization Module
│   ├── __init__.py
│   ├── scene_manager.py      # Scene management:
│   │                         #   • Mesh rendering
│   │                         #   • Camera control
│   │                         #   • Scene bounds computation
│   │                         #   • Model focus
│   └── clipping.py           # Clipping planes:
│                             #   • Plane creation and management
│                             #   • Plane visualization
│                             #   • Slice synchronization
│
├── slices/                   # Slice Viewers Module
│   ├── __init__.py
│   ├── viewer.py             # 2D slice display:
│   │                         #   • Slice extraction from NIFTI data
│   │                         #   • Mask overlay
│   │                         #   • MPR point marking overlay
│   └── slice_3d.py           # 3D slice visualization:
│                             #   • Perpendicular slice images in 3D scene
│                             #   • Slice transform management
│
├── mpr/                      # Multi-Planar Reconstruction Module
│   ├── __init__.py
│   └── handlers.py           # Curved MPR functionality:
│                             #   • Point marking on axial slices
│                             #   • Spline-based path generation
│                             #   • MPR generation from marked points
│                             #   • MPR result display
│
└── gltf/                     # GLTF/GLB Handlers Module
    ├── __init__.py
    └── handler.py            # GLTF/GLB model handling:
                              #   • Static model import (via trimesh)
                              #   • Animated model viewer (via Three.js)
```

## 🏗️ Architecture

This project follows a **modular architecture** with clear separation of concerns:

| Module | Responsibility |
|--------|----------------|
| **UI Layer** | Handles all user interface components and styling |
| **Loaders** | Manages file I/O for various medical imaging and mesh formats |
| **Visualization** | Core 3D rendering and scene management |
| **Slices** | 2D slice extraction and display logic |
| **MPR** | Multi-planar reconstruction algorithms and path generation |
| **GLTF** | Specialized handling for GLTF/GLB model formats |

### Benefits of Modular Design

1. **Maintainability**: Each module has a single, well-defined responsibility
2. **Testability**: Individual modules can be tested independently
3. **Reusability**: Functions can be reused across different parts of the application
4. **Scalability**: Easy to add new features without bloating existing code
5. **Readability**: Smaller, focused files are easier to understand and navigate

## 🔧 Development

### Adding New Features

The modular structure makes it easy to extend functionality:

- **New file formats**: Add handlers to `loaders/file_handlers.py`
- **New visualization features**: Extend `visualization/scene_manager.py`
- **New UI components**: Add widget creation functions to `ui/widgets.py`
- **New algorithms**: Create new modules following the existing structure

### Code Style

- Follow PEP 8 Python style guidelines
- Use descriptive function and variable names
- Include docstrings for all public functions and classes
- Keep functions focused and single-purpose

## 📝 Notes

- The original `cmpr_gui_v2.py` file is kept for reference. All functionality has been preserved in the modular structure.
- The `STLViewer` class in `main.py` maintains the same interface, so existing scripts importing from `cmpr_core_v2` would need to be updated to import from `main` instead.
- Application settings are persisted using QSettings and stored per-user on the system.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[Specify your license here - MIT, Apache 2.0, etc.]

## 🙏 Acknowledgments

This project is built using the following excellent open-source libraries:

- **[PyQt5](https://www.riverbankcomputing.com/software/pyqt/)** - GUI framework
- **[VisPy](https://vispy.org/)** - High-performance 3D visualization
- **[Nibabel](https://nipy.org/nibabel/)** - Medical imaging support
- **[Trimesh](https://trimsh.org/)** - 3D mesh processing

---

**Note**: This application is for research and educational purposes. Always verify medical imaging data with appropriate clinical tools.
