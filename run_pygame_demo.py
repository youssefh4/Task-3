import vtk
import numpy as np
import random
import math
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
import tkinter as tk
from tkinter import filedialog
import os
import sys

class BrainElectricViewer:
    def __init__(self):
        # Initialize Tkinter for file dialog
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        style = vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        self.renderer.SetBackground(0.1, 0.1, 0.15)  # Dark blue background
        # Improve translucent rendering for better visibility of effects
        try:
            self.render_window.SetMultiSamples(0)
            self.renderer.SetUseDepthPeeling(True)
            self.renderer.SetMaximumNumberOfPeels(100)
            self.renderer.SetOcclusionRatio(0.1)
        except Exception:
            pass
        self.electric_points = []
        self.electric_actors = []
        self.timer_count = 0
        # Particle-style storage for animated bolts
        self.bolts = []  # each: { 'line': vtkLineSource, 'actor': vtkActor, 'age': int, 'max_age': int, 'velocity': np.ndarray }
        # Global intensity multiplier controlled by keyboard (+/-)
        self.intensity_scale = 1.0
        # Dynamic modulation controls
        self.dynamic_enabled = True
        self.dynamic_phase = 0.0
        self.dynamic_speed = 0.25  # radians per frame
        
        # Set window title
        self.render_window.SetWindowName("Brain Electric Visualization - User: youssefh4")

    def select_files(self):
        """Open file dialog to select one or more 3D model files"""
        file_paths = filedialog.askopenfilenames(
            title="Select 3D Brain Model(s)",
            filetypes=[
                ("3D Models", "*.obj *.stl *.ply"),
                ("OBJ files", "*.obj"),
                ("STL files", "*.stl"),
                ("PLY files", "*.ply"),
                ("All files", "*.*")
            ]
        )
        
        if not file_paths:
            print("No files selected. Exiting...")
            sys.exit(0)
            
        return list(file_paths)
        
    def load_models(self, file_paths=None):
        if file_paths is None:
            file_paths = self.select_files()
        
        # Storage for multiple models
        self.models_data = []
        self.model_actors = []
        
        try:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    print(f"Error: File '{file_path}' not found. Skipping...")
                    continue
                
                extension = file_path.lower().split('.')[-1]
                if extension == 'obj':
                    reader = vtk.vtkOBJReader()
                elif extension == 'stl':
                    reader = vtk.vtkSTLReader()
                elif extension == 'ply':
                    reader = vtk.vtkPLYReader()
                else:
                    print(f"Error: Unsupported file format: {extension}. Skipping {file_path}")
                    continue
                
                reader.SetFileName(file_path)
                reader.Update()
                poly = reader.GetOutput()
                
                if poly.GetNumberOfPoints() == 0:
                    print(f"Warning: '{os.path.basename(file_path)}' has no points. Skipping...")
                    continue
                
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly)
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # Brain-like soft tissue color (warm pinkish tone)
                # Slight variation per model to avoid flat look
                tint = 0.03 * (random.random() - 0.5)
                r, g, b = 0.95 + tint, 0.80 + tint, 0.85 + tint
                r = max(0.0, min(1.0, r)); g = max(0.0, min(1.0, g)); b = max(0.0, min(1.0, b))
                actor.GetProperty().SetColor(r, g, b)
                # Semi-translucent to see internal effects
                actor.GetProperty().SetOpacity(0.45)
                # Softer, more organic shading
                actor.GetProperty().SetAmbient(0.35)
                actor.GetProperty().SetDiffuse(0.55)
                actor.GetProperty().SetSpecular(0.25)
                actor.GetProperty().SetSpecularPower(8)
                
                self.renderer.AddActor(actor)
                self.models_data.append(poly)
                self.model_actors.append(actor)
                
                print(f"Loaded model: {os.path.basename(file_path)} | Points: {poly.GetNumberOfPoints()} | Polys: {poly.GetNumberOfPolys()}")
            
            if not self.models_data:
                print("No valid models were loaded. Exiting...")
                sys.exit(1)
            
            self.renderer.ResetCamera()
            
            # Initialize electric effect across all loaded models
            self.setup_electric_effect()
        
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            sys.exit(1)

    def setup_electric_effect(self):
        # Aggregate points from all loaded models
        self.surface_points = []
        total_samples = 50
        if not hasattr(self, 'models_data') or not self.models_data:
            print("No models to sample for electric effect.")
            return
        
        # Distribute samples across models
        samples_per_model = max(1, total_samples // len(self.models_data))
        for poly in self.models_data:
            pts = poly.GetPoints()
            if pts is None:
                continue
            n_points = pts.GetNumberOfPoints()
            if n_points == 0:
                continue
            for _ in range(samples_per_model):
                idx = random.randint(0, n_points - 1)
                self.surface_points.append(pts.GetPoint(idx))
        
        # If we didn't reach total_samples due to few models/points, pad randomly from first model
        if len(self.surface_points) < total_samples:
            first_pts = self.models_data[0].GetPoints()
            if first_pts is not None and first_pts.GetNumberOfPoints() > 0:
                needed = total_samples - len(self.surface_points)
                for _ in range(needed):
                    idx = random.randint(0, first_pts.GetNumberOfPoints() - 1)
                    self.surface_points.append(first_pts.GetPoint(idx))
        
        print(f"Electric effect initialized with {len(self.surface_points)} points across {len(self.models_data)} model(s)")

    def create_electric_line(self, start_point, end_point, intensity=1.0):
        """Create a glowing electric line between two points"""
        # Create line
        line = vtk.vtkLineSource()
        line.SetPoint1(start_point)
        line.SetPoint2(end_point)
        
        # Create tube around line
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputConnection(line.GetOutputPort())
        tube_filter.SetRadius(0.05)  # Thicker tubes for visibility
        tube_filter.SetNumberOfSides(8)
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        
        # Create actor with glowing effect
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set electric glow properties
        # Electric: vibrant blue with slight cyan tint
        actor.GetProperty().SetColor(0.2, 0.6, 1.0)
        # Make it appear emissive/unlit so it stands out
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.GetProperty().SetSpecular(0.0)
        actor.GetProperty().SetSpecularPower(1)
        actor.GetProperty().SetOpacity(max(0.05, min(1.0, float(intensity))))
        
        return actor, line

    def update_electric_effect(self, obj, event):
        """Update the electric effect animation with fading/motion"""
        # Parameters
        max_bolts = 150
        spawn_per_frame = 8
        max_age_frames = 36  # longer life for clearer fade
        drift_scale = 0.02

        # Dynamic modulation: waves of activity
        if self.dynamic_enabled:
            self.dynamic_phase += self.dynamic_speed
            dyn = 0.5 + 0.5 * math.sin(self.dynamic_phase)
            spawn_per_frame = max(1, int(spawn_per_frame * (0.7 + 0.8 * dyn)))
            max_bolts = max(50, int(max_bolts * (0.8 + 0.5 * dyn)))
            drift_scale = drift_scale * (0.7 + 0.6 * dyn)

        # Age and update existing bolts
        survivors = []
        for bolt in self.bolts:
            bolt['age'] += 1
            age = bolt['age']
            max_age = bolt['max_age']
            # Envelope: smooth fade in/out using sin(pi * t)
            t = min(1.0, max(0.0, age / max_age))
            life_alpha = math.sin(math.pi * t)
            # Pulsing flicker with per-bolt speed and phase
            omega = bolt.get('omega', 0.25)
            pulse = 0.5 * (1.0 + math.sin(self.timer_count * omega + bolt.get('phase', 0.0)))
            # Combine envelope and pulse for moving electricity look
            combined_alpha = (0.2 + 0.8 * pulse) * life_alpha
            # Apply global intensity multiplier
            combined_alpha *= self.intensity_scale
            combined_alpha = max(0.05, min(1.0, combined_alpha))
            bolt['actor'].GetProperty().SetOpacity(combined_alpha)

            # Slightly move end along velocity to suggest motion
            try:
                p1 = np.array(bolt['line'].GetPoint1())
                p2 = np.array(bolt['line'].GetPoint2())
                v = bolt['velocity']
                p1 = p1 + v * drift_scale
                p2 = p2 + v * drift_scale
                bolt['line'].SetPoint1(p1)
                bolt['line'].SetPoint2(p2)
                bolt['line'].Modified()
            except Exception:
                pass

            if age < max_age:
                survivors.append(bolt)
            else:
                # Remove expired bolt from scene
                try:
                    self.renderer.RemoveActor(bolt['actor'])
                except Exception:
                    pass

        self.bolts = survivors

        # Spawn new bolts
        if hasattr(self, 'surface_points') and self.surface_points:
            num_to_spawn = min(spawn_per_frame, max(0, max_bolts - len(self.bolts)))
            for _ in range(num_to_spawn):
                start_point = random.choice(self.surface_points)
                # Random outward/inward direction
                direction = np.random.normal(size=3)
                norm = np.linalg.norm(direction) or 1.0
                direction = direction / norm
                # Length can also breathe with dynamics
                if self.dynamic_enabled:
                    dyn_len = 0.5 + 0.5 * math.sin(self.dynamic_phase + random.random())
                    len_min, len_max = 0.15, 0.9
                    length = random.uniform(len_min, len_max) * (0.8 + 0.6 * dyn_len)
                else:
                    length = random.uniform(0.2, 0.8)
                end_point = np.array(start_point) + direction * length

                intensity = random.uniform(0.7, 1.0)
                actor, line = self.create_electric_line(start_point, tuple(end_point), intensity)
                self.renderer.AddActor(actor)

                bolt = {
                    'line': line,
                    'actor': actor,
                    'age': 0,
                    'max_age': random.randint(int(max_age_frames * 0.6), max_age_frames),
                    'velocity': direction * random.uniform(0.5, 1.5),
                    'phase': random.uniform(0.0, 2.0 * math.pi),
                    'omega': random.uniform(0.18, 0.42)  # pulse speed
                }
                self.bolts.append(bolt)

        self.timer_count += 1
        self.render_window.Render()

    def add_keyboard_controls(self):
        """Add keyboard controls for interaction"""
        def handle_key(obj, event):
            key = obj.GetKeySym()
            if key == "escape":
                self.interactor.TerminateApp()
            elif key == "plus" or key == "equal":
                # Increase global intensity multiplier
                self.intensity_scale = min(3.0, self.intensity_scale + 0.1)
                print(f"Intensity x{self.intensity_scale:.2f}")
            elif key == "minus":
                # Decrease global intensity multiplier
                self.intensity_scale = max(0.1, self.intensity_scale - 0.1)
                print(f"Intensity x{self.intensity_scale:.2f}")
            elif key == "d":
                # Toggle dynamic modulation
                self.dynamic_enabled = not self.dynamic_enabled
                state = "ON" if self.dynamic_enabled else "OFF"
                print(f"Dynamic modulation: {state}")
            elif key == "bracketleft":
                # Slow down dynamic speed
                self.dynamic_speed = max(0.05, self.dynamic_speed - 0.05)
                print(f"Dynamic speed: {self.dynamic_speed:.2f}")
            elif key == "bracketright":
                # Speed up dynamic speed
                self.dynamic_speed = min(1.00, self.dynamic_speed + 0.05)
                print(f"Dynamic speed: {self.dynamic_speed:.2f}")
            elif key == "r":
                # Reset effect
                for bolt in self.bolts:
                    try:
                        self.renderer.RemoveActor(bolt['actor'])
                    except Exception:
                        pass
                self.bolts.clear()
                self.dynamic_phase = 0.0
                print("Effect reset")
        
        self.interactor.AddObserver("KeyPressEvent", handle_key)

    def start(self):
        print("\nControls:")
        print("- Left mouse button: Rotate")
        print("- Middle mouse button: Pan")
        print("- Right mouse button: Zoom")
        print("- Plus/Minus keys: Adjust electric effect intensity")
        print("- ESC: Exit")
        
        # Add keyboard controls
        self.add_keyboard_controls()
        
        # Add observer for animation
        self.interactor.AddObserver('TimerEvent', self.update_electric_effect)
        self.interactor.CreateRepeatingTimer(50)  # Update every 50ms
        
        # Start visualization
        self.render_window.Render()
        self.interactor.Start()

def main():
    print("Brain Electric Visualization")
    print("Current Date/Time (UTC):", "2025-11-04 13:34:27")
    print("User:", "youssefh4")
    
    viewer = BrainElectricViewer()
    
    # Check if file paths were provided as command line arguments
    if len(sys.argv) > 1:
        # File paths provided via command line
        file_paths = sys.argv[1:]
        print(f"\nLoading {len(file_paths)} brain model file(s) from command line...")
        for path in file_paths:
            print(f"  - {path}")
        viewer.load_models(file_paths=file_paths)
    else:
        # No arguments provided, prompt for file selection
        print("\nPlease select your 3D brain model file...")
        viewer.load_models()  # Will prompt for multiple file selection
    
    viewer.start()

if __name__ == "__main__":
    main()