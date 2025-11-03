# 3D Medical Image Viewer - Refactored Structure

This project has been refactored from a single large file (`cmpr_core_v2.py`) into a clean, modular structure for better maintainability.

## Project Structure

```
Task 3/
├── main.py                    # Main application entry point
├── cmpr_core_v2.py           # Original file (kept for reference)
│
├── ui/                       # UI Components
│   ├── __init__.py
│   ├── styles.py             # UI styling constants
│   └── widgets.py            # Widget creation functions
│
├── loaders/                  # File Loading
│   ├── __init__.py
│   └── file_handlers.py      # Mesh, NIFTI, and mask loading
│
├── visualization/            # 3D Visualization
│   ├── __init__.py
│   ├── scene_manager.py      # Scene management and mesh rendering
│   └── clipping.py           # Clipping planes functionality
│
├── slices/                   # Slice Viewers
│   ├── __init__.py
│   ├── viewer.py             # 2D slice display logic
│   └── slice_3d.py           # 3D slice visualization
│
├── mpr/                      # MPR (Multi-Planar Reconstruction)
│   ├── __init__.py
│   └── handlers.py           # MPR point marking and generation
│
└── gltf/                     # GLTF/GLB Handlers
    ├── __init__.py
    └── handler.py            # GLTF import and animated viewer
```

## Module Descriptions

### `ui/` - User Interface Components
- **styles.py**: Contains application-wide styling constants
- **widgets.py**: Functions to create UI widgets (buttons, group boxes, sliders, etc.)

### `loaders/` - File Loading
- **file_handlers.py**: Functions for loading:
  - 3D mesh files (STL, OBJ, GLB, GLTF)
  - NIFTI medical images
  - Segmentation masks

### `visualization/` - 3D Visualization
- **scene_manager.py**: Manages 3D scene:
  - Mesh rendering
  - Camera control
  - Scene bounds computation
  - Model focus
- **clipping.py**: Clipping planes:
  - Plane creation and management
  - Plane visualization
  - Slice synchronization

### `slices/` - Slice Viewers
- **viewer.py**: 2D slice display:
  - Slice extraction from NIFTI data
  - Mask overlay
  - MPR point marking overlay
- **slice_3d.py**: 3D slice visualization:
  - Perpendicular slice images in 3D scene
  - Slice transform management

### `mpr/` - Multi-Planar Reconstruction
- **handlers.py**: Curved MPR functionality:
  - Point marking on axial slices
  - MPR generation from marked points
  - MPR result display

### `gltf/` - GLTF/GLB Support
- **handler.py**: GLTF/GLB model handling:
  - Static model import (via trimesh)
  - Animated model viewer (via Three.js in QWebEngine)

## Running the Application

Run the main application:
```bash
python main.py
```

## Benefits of Refactoring

1. **Better Organization**: Code is now organized by functionality
2. **Easier Maintenance**: Each module has a single responsibility
3. **Improved Readability**: Smaller, focused files are easier to understand
4. **Better Testing**: Individual modules can be tested independently
5. **Reusability**: Functions can be reused across different parts of the application
6. **Scalability**: Easy to add new features without bloating a single file

## Migration Notes

The original `cmpr_core_v2.py` file is kept for reference. All functionality has been preserved in the modular structure. The `STLViewer` class in `main.py` maintains the same interface, so existing scripts that import from `cmpr_core_v2` would need to be updated to import from `main` instead.
