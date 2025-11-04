"""File loading handlers for meshes, NIFTI, and masks."""

import os
import numpy as np
import trimesh
import nibabel as nib
from PyQt5 import QtWidgets, QtCore
from skimage import measure


def load_mesh_folder(viewer_instance):
    """Load mesh files from a folder.
    
    Args:
        viewer_instance: STLViewer instance with meshes, model_list, etc.
    """
    folder = QtWidgets.QFileDialog.getExistingDirectory(
        viewer_instance, "Select Folder with Mesh Files"
    )
    if not folder:
        return
    
    viewer_instance.model_list.clear()
    viewer_instance.model_dropdown.clear()
    viewer_instance.meshes.clear()
    if hasattr(viewer_instance, 'mesh_file_paths'):
        viewer_instance.mesh_file_paths.clear()
    viewer_instance.clear_scene()
    viewer_instance.initial_camera_state = None
    viewer_instance.scene_bounds = None
    viewer_instance.focused_model = None
    viewer_instance.model_dropdown.setEnabled(False)
    
    viewer_instance.model_dropdown.addItem("None")
    viewer_instance.model_dropdown.setCurrentIndex(0)
    
    mesh_files = [
        f for f in os.listdir(folder) 
        if f.lower().endswith((".stl", ".obj", ".glb", ".gltf"))
    ]
    if not mesh_files:
        QtWidgets.QMessageBox.warning(
            viewer_instance, "No Files", 
            "No STL, OBJ, GLB, or GLTF files found."
        )
        return
    
    for f in mesh_files:
        path = os.path.join(folder, f)
        mesh = trimesh.load_mesh(path, force="mesh")
        if mesh.is_empty:
            continue
        print(f"Loaded: {f} | Vertices: {len(mesh.vertices)} Faces: {len(mesh.faces)}")
        viewer_instance.meshes[f] = mesh
        # Store the original file path
        if not hasattr(viewer_instance, 'mesh_file_paths'):
            viewer_instance.mesh_file_paths = {}
        viewer_instance.mesh_file_paths[f] = path
        item = QtWidgets.QListWidgetItem(f)
        item.setCheckState(QtCore.Qt.Checked)
        viewer_instance.model_list.addItem(item)
        viewer_instance.model_dropdown.addItem(f)
    
    viewer_instance.model_dropdown.setEnabled(len(viewer_instance.meshes) > 0)
    viewer_instance.model_dropdown.setCurrentIndex(0)
    viewer_instance.update_scene()


def load_nifti_file(viewer_instance):
    """Load NIFTI file and setup slice viewers.
    
    Args:
        viewer_instance: STLViewer instance with nifti_data, slice_sliders, etc.
    """
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        viewer_instance, "Select NIFTI File", "", "NIFTI files (*.nii *.nii.gz)"
    )
    if not file_path:
        return
    
    try:
        # Load NIFTI file
        nifti = nib.load(file_path)
        viewer_instance.nifti_data = nifti.get_fdata()
        viewer_instance.nifti_shape = viewer_instance.nifti_data.shape
        viewer_instance.nifti_affine = nifti.affine
        
        print(f"Loaded NIFTI: shape={viewer_instance.nifti_shape}")
        
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            viewer_instance, "Error", f"Failed to load NIFTI file:\n{str(e)}"
        )
        return
    
    try:
        # Update sliders min/max values based on image dimensions
        from slices.viewer import setup_slice_sliders, connect_slice_sliders
        
        setup_slice_sliders(viewer_instance)
        connect_slice_sliders(viewer_instance)
        
        # Display initial slices
        from slices.viewer import update_slice_display
        update_slice_display(viewer_instance, 'axial', 
                            viewer_instance.slice_sliders['axial'].value())
        update_slice_display(viewer_instance, 'sagittal',
                            viewer_instance.slice_sliders['sagittal'].value())
        update_slice_display(viewer_instance, 'coronal',
                            viewer_instance.slice_sliders['coronal'].value())
        
        # MPR is now handled in a separate window
        # No need to enable buttons here
        
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            viewer_instance, "Error", f"Failed to setup slice viewers:\n{str(e)}"
        )


def load_dicom_folder(folder_path):
    """Load DICOM series from a folder and convert to 3D numpy array.
    
    Args:
        folder_path: Path to folder containing DICOM files
        
    Returns:
        tuple: (3D numpy array, affine matrix)
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM support. Install with: pip install pydicom"
        )
    
    # Find all DICOM files in the folder (recursively)
    dicom_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Try to read as DICOM (will fail if not DICOM)
                ds = pydicom.dcmread(file_path, force=True)
                if hasattr(ds, 'SliceLocation') or hasattr(ds, 'ImagePositionPatient'):
                    dicom_files.append((file_path, ds))
            except:
                continue
    
    if not dicom_files:
        raise ValueError("No DICOM files found in the selected folder")
    
    # Sort by slice location or instance number
    try:
        # Try to sort by SliceLocation
        dicom_files.sort(key=lambda x: float(x[1].SliceLocation) if hasattr(x[1], 'SliceLocation') and x[1].SliceLocation != '' else 0)
    except:
        try:
            # Fallback to InstanceNumber
            dicom_files.sort(key=lambda x: int(x[1].InstanceNumber) if hasattr(x[1], 'InstanceNumber') and x[1].InstanceNumber != '' else 0)
        except:
            # Fallback to filename
            dicom_files.sort(key=lambda x: x[0])
    
    # Load pixel arrays
    slices = []
    for file_path, ds in dicom_files:
        try:
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Apply rescale slope and intercept if available
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            
            slices.append(pixel_array)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue
    
    if not slices:
        raise ValueError("No valid DICOM slices could be loaded")
    
    # Stack slices into 3D array
    volume = np.stack(slices, axis=0)
    
    # Create affine matrix from DICOM metadata
    # Use identity matrix as default, but try to get spacing/orientation if available
    affine = np.eye(4)
    
    if len(dicom_files) > 0:
        first_ds = dicom_files[0][1]
        
        # Get pixel spacing
        if hasattr(first_ds, 'PixelSpacing') and first_ds.PixelSpacing:
            pixel_spacing = [float(x) for x in first_ds.PixelSpacing]
            affine[0, 0] = pixel_spacing[0] if len(pixel_spacing) > 0 else 1.0
            affine[1, 1] = pixel_spacing[1] if len(pixel_spacing) > 1 else 1.0
        
        # Get slice thickness
        if hasattr(first_ds, 'SliceThickness') and first_ds.SliceThickness:
            try:
                slice_thickness = float(first_ds.SliceThickness)
                affine[2, 2] = slice_thickness
            except:
                pass
        
        # Get image position if available
        if hasattr(first_ds, 'ImagePositionPatient') and first_ds.ImagePositionPatient:
            try:
                position = [float(x) for x in first_ds.ImagePositionPatient]
                affine[0:3, 3] = position[:3]
            except:
                pass
    
    print(f"Loaded {len(slices)} DICOM slices, shape: {volume.shape}")
    return volume, affine


def load_masks_folder(viewer_instance):
    """Load mask files from a segmentation folder.
    
    Args:
        viewer_instance: STLViewer instance with nifti_data, mask_data, etc.
    """
    if viewer_instance.nifti_data is None:
        QtWidgets.QMessageBox.warning(
            viewer_instance, "No NIFTI Loaded", "Please load a NIFTI file first."
        )
        return
    
    folder = QtWidgets.QFileDialog.getExistingDirectory(
        viewer_instance, "Select Segmentation Folder"
    )
    if not folder:
        return
    
    try:
        # Find all NIFTI files in the folder
        mask_files = [
            f for f in os.listdir(folder) 
            if f.lower().endswith((".nii", ".nii.gz"))
        ]
        if not mask_files:
            QtWidgets.QMessageBox.warning(
                viewer_instance, "No Masks",
                "No NIFTI mask files found in the selected folder."
            )
            return
        
        # Clear previous masks
        viewer_instance.mask_data = []
        viewer_instance.mask_files = []
        viewer_instance.mask_colors = []
        viewer_instance._slices_with_masks_cache.clear()
        
        # Load each mask file
        for mask_file in mask_files:
            path = os.path.join(folder, mask_file)
            nifti = nib.load(path)
            mask_data = nifti.get_fdata()
            
            # Check if dimensions match
            if mask_data.shape != viewer_instance.nifti_shape:
                QtWidgets.QMessageBox.warning(
                    viewer_instance, "Dimension Mismatch",
                    f"Mask {mask_file} has shape {mask_data.shape}, "
                    f"expected {viewer_instance.nifti_shape}"
                )
                continue
            
            viewer_instance.mask_data.append(mask_data)
            viewer_instance.mask_files.append(mask_file)
            # Generate a random color for each mask
            viewer_instance.mask_colors.append(np.random.rand(3))
        
        print(f"Loaded {len(viewer_instance.mask_data)} mask(s)")
        
        # Clear cache for slices with masks
        viewer_instance._slices_with_masks_cache.clear()
        
        # Update slice views to show masks
        from slices.viewer import update_slice_display
        update_slice_display(viewer_instance, 'axial',
                            viewer_instance.slice_sliders['axial'].value())
        update_slice_display(viewer_instance, 'sagittal',
                            viewer_instance.slice_sliders['sagittal'].value())
        update_slice_display(viewer_instance, 'coronal',
                            viewer_instance.slice_sliders['coronal'].value())
        
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            viewer_instance, "Error", f"Failed to load masks:\n{str(e)}"
        )


def visualize_masks_3d(viewer_instance):
    """Convert loaded masks to 3D meshes and display in the 3D viewer.
    
    Args:
        viewer_instance: STLViewer instance with mask_data, meshes, etc.
    """
    if not viewer_instance.mask_data:
        QtWidgets.QMessageBox.warning(
            viewer_instance, "No Masks", "Please load masks first."
        )
        return
    
    try:
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.WaitCursor
        )
        
        # Remove any existing mask-derived meshes (those with "_mask" suffix)
        items_to_remove = []
        for i in range(viewer_instance.model_list.count()):
            item = viewer_instance.model_list.item(i)
            if item.text().endswith("_mask"):
                items_to_remove.append(item.text())
        
        for mesh_name in items_to_remove:
            if mesh_name in viewer_instance.meshes:
                del viewer_instance.meshes[mesh_name]
            if mesh_name in viewer_instance.mesh_colors:
                del viewer_instance.mesh_colors[mesh_name]
            # Remove from list
            for i in range(viewer_instance.model_list.count()):
                if viewer_instance.model_list.item(i).text() == mesh_name:
                    viewer_instance.model_list.takeItem(i)
                    break
            # Remove from dropdown
            index = viewer_instance.model_dropdown.findText(mesh_name)
            if index >= 0:
                viewer_instance.model_dropdown.removeItem(index)
        
        # Convert each mask to a 3D mesh
        for mask_idx, (mask, mask_file, color) in enumerate(
            zip(viewer_instance.mask_data, viewer_instance.mask_files, 
                viewer_instance.mask_colors)
        ):
            # Create mesh name from file
            mesh_name = os.path.splitext(os.path.splitext(mask_file)[0])[0] + "_mask"
            
            # Threshold the mask
            binary_mask = mask > 0.5
            
            # Use marching cubes to create 3D mesh
            try:
                verts, faces, normals, values = measure.marching_cubes(
                    binary_mask.astype(float),
                    level=0.5,
                    spacing=(1.0, 1.0, 1.0),
                    method='lorensen'
                )
                
                # Create trimesh from the marching cubes result
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                
                if not mesh.is_empty:
                    print(f"Created 3D mesh from {mask_file}: "
                          f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                    
                    # Store the mesh
                    viewer_instance.meshes[mesh_name] = mesh
                    viewer_instance.mesh_colors[mesh_name] = color
                    
                    # Add to list with checkbox checked
                    item = QtWidgets.QListWidgetItem(mesh_name)
                    item.setCheckState(QtCore.Qt.Checked)
                    viewer_instance.model_list.addItem(item)
                    
                    # Add to dropdown
                    viewer_instance.model_dropdown.addItem(mesh_name)
            except Exception as e:
                print(f"Failed to create mesh from {mask_file}: {e}")
                continue
        
        # Enable dropdown if we have meshes
        if len(viewer_instance.meshes) > 0:
            viewer_instance.model_dropdown.setEnabled(True)
        
        # Update the 3D scene to show the new meshes
        viewer_instance.update_scene()
        
        QtWidgets.QApplication.restoreOverrideCursor()
        print(f"Converted {len([m for m in viewer_instance.meshes.keys() if m.endswith('_mask')])} "
              f"mask(s) to 3D meshes")
        
    except Exception as e:
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.critical(
            viewer_instance, "Error",
            f"Failed to visualize masks in 3D:\n{str(e)}"
        )
        import traceback
        traceback.print_exc()

