"""GLTF/GLB model import and animated viewer."""

import os
import base64
import trimesh
from PyQt5 import QtWidgets, QtCore


try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except ImportError:
    WEBENGINE_AVAILABLE = False


def import_gltf_model(viewer_instance):
    """Import a GLTF/GLB model as static geometry using trimesh."""
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        viewer_instance,
        "Select GLTF/GLB Model",
        os.path.expanduser("~"),
        "GLTF/GLB Files (*.glb *.gltf)"
    )
    if not file_path:
        return
    
    try:
        mesh = trimesh.load_mesh(file_path, force="mesh")
        if mesh.is_empty:
            QtWidgets.QMessageBox.warning(
                viewer_instance, "GLTF Import", "Loaded geometry is empty."
            )
            return
        
        name = os.path.basename(file_path)
        viewer_instance.meshes[name] = mesh
        item = QtWidgets.QListWidgetItem(name)
        item.setCheckState(QtCore.Qt.Checked)
        viewer_instance.model_list.addItem(item)
        viewer_instance.model_controls['dropdown'].addItem(name)
        viewer_instance.model_controls['dropdown'].setEnabled(True)
        viewer_instance.update_scene()
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            viewer_instance, "GLTF Import",
            f"Failed to import GLTF/GLB file:\n{str(e)}"
        )


def import_gltf_animated(viewer_instance):
    """Open a GLTF/GLB model in a Three.js viewer with full animation support."""
    if not WEBENGINE_AVAILABLE:
        QtWidgets.QMessageBox.warning(
            viewer_instance,
            "WebEngine Not Available",
            "PyQt5.QtWebEngineWidgets is not installed.\n"
            "Please install it with: pip install PyQtWebEngine"
        )
        return
    
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        viewer_instance,
        "Select GLTF/GLB Model (with animations)",
        os.path.expanduser("~"),
        "GLTF/GLB Files (*.glb *.gltf)"
    )
    if not file_path:
        return
    
    try:
        # Read file and convert to base64 data URL
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        file_ext = os.path.splitext(file_path)[1].lower()
        encoded = None
        
        if file_ext == '.glb':
            # Binary GLB file
            mime_type = 'model/gltf-binary'
            encoded = base64.b64encode(file_data).decode('utf-8')
            file_url = f'data:{mime_type};base64,{encoded}'
        else:
            # GLTF JSON file
            file_url = os.path.abspath(file_path).replace('\\', '/')
            if not file_url.startswith('/'):
                file_url = '/' + file_url
            file_url = 'file://' + file_url
        
        # Generate HTML with Three.js
        html_content = _generate_threejs_viewer_html(file_url, file_path)
        
        # Create window with QWebEngineView
        window = QtWidgets.QMainWindow(viewer_instance)
        window.setWindowTitle(f"GLTF Viewer: {os.path.basename(file_path)}")
        window.setGeometry(100, 100, 1200, 800)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        
        # Central widget
        central_widget = QtWidgets.QWidget()
        window.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create web view
        web_view = QWebEngineView()
        
        # Set up error handling
        def handle_js_console_message(level, message, line_number, source_id):
            print(f"JS Console [{level}]: {message} "
                  f"(line {line_number}, source: {source_id})")
        
        try:
            web_view.page().javaScriptConsoleMessage.connect(handle_js_console_message)
        except AttributeError:
            print("Note: JavaScript console message handler not available")
        
        # Set HTML content
        try:
            web_view.setHtml(html_content)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                viewer_instance, "HTML Error",
                f"Failed to set HTML content:\n{str(e)}"
            )
            return
        
        layout.addWidget(web_view)
        
        # Add controls toolbar
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        play_btn = QtWidgets.QPushButton("▶️ Play")
        pause_btn = QtWidgets.QPushButton("⏸️ Pause")
        stop_btn = QtWidgets.QPushButton("⏹️ Stop")
        
        def play_anim():
            try:
                web_view.page().runJavaScript("if (mixer) mixer.playAll();")
            except Exception as e:
                print(f"Error playing animation: {e}")
        
        def pause_anim():
            try:
                web_view.page().runJavaScript("if (mixer) mixer.pauseAll();")
            except Exception as e:
                print(f"Error pausing animation: {e}")
        
        def stop_anim():
            try:
                web_view.page().runJavaScript(
                    "if (mixer) mixer.stopAll(); if (clock) clock.elapsedTime = 0;"
                )
            except Exception as e:
                print(f"Error stopping animation: {e}")
        
        play_btn.clicked.connect(play_anim)
        pause_btn.clicked.connect(pause_anim)
        stop_btn.clicked.connect(stop_anim)
        
        controls_layout.addWidget(play_btn)
        controls_layout.addWidget(pause_btn)
        controls_layout.addWidget(stop_btn)
        controls_layout.addStretch()
        
        layout.addWidget(controls_widget)
        
        # Store window reference
        viewer_instance.gltf_windows.append(window)
        
        # Clean up reference when window closes
        def closeEvent(event):
            try:
                if window in viewer_instance.gltf_windows:
                    viewer_instance.gltf_windows.remove(window)
                    print("GLTF viewer window closed and removed from references")
            except Exception as e:
                print(f"Error in window close handler: {e}")
            event.accept()
        
        window.closeEvent = closeEvent
        
        # Show window
        try:
            window.show()
            window.raise_()
            window.activateWindow()
            print(f"GLTF viewer window opened for: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error showing window: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            viewer_instance, "GLTF Viewer Error",
            f"Failed to open GLTF viewer:\n{str(e)}"
        )
        import traceback
        traceback.print_exc()


def _generate_threejs_viewer_html(file_url, file_path):
    """Generate HTML content for Three.js GLTF viewer with animations."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GLTF Viewer</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: #222;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: Arial;
            font-size: 12px;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div id="info">
        <div>Drag to rotate • Scroll to zoom • Right-click to pan</div>
        <div id="anim-info">Loading...</div>
    </div>
    <script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        let scene, camera, renderer, controls;
        let mixer, clock;
        let model = null;
        
        // Initialize Three.js
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x222222);
        
        camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 1, 3);
        
        renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        document.body.appendChild(renderer.domElement);
        
        // Orbit controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 5);
        directionalLight.castShadow = true;
        scene.add(directionalLight);
        
        const pointLight = new THREE.PointLight(0xffffff, 0.4);
        pointLight.position.set(-5, 5, -5);
        scene.add(pointLight);
        
        // Animation clock
        clock = new THREE.Clock();
        mixer = null;
        
        // Load GLTF/GLB
        const loader = new THREE.GLTFLoader();
        
        loader.load(
            '{file_url}',
            function (gltf) {{
                model = gltf.scene;
                scene.add(model);
                
                // Center and scale model
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 2.0 / maxDim;
                model.scale.multiplyScalar(scale);
                
                box.setFromObject(model);
                box.getCenter(center);
                box.getSize(size);
                
                model.position.x += (model.position.x - center.x);
                model.position.y += (model.position.y - center.y);
                model.position.z += (model.position.z - center.z);
                
                // Set up animations
                if (gltf.animations && gltf.animations.length > 0) {{
                    mixer = new THREE.AnimationMixer(model);
                    
                    gltf.animations.forEach((clip) => {{
                        const action = mixer.clipAction(clip);
                        action.play();
                    }});
                    
                    document.getElementById('anim-info').innerHTML = 
                        `Animations: ${{gltf.animations.length}} | Playing: ${{gltf.animations.length}} clip(s)`;
                }} else {{
                    document.getElementById('anim-info').innerHTML = 'No animations found';
                }}
                
                // Update camera to fit model
                box.setFromObject(model);
                box.getCenter(center);
                box.getSize(size);
                
                const maxSize = Math.max(size.x, size.y, size.z);
                const distance = maxSize * 2;
                camera.position.set(distance, distance, distance);
                camera.lookAt(center);
                controls.target.copy(center);
                controls.update();
            }},
            function (xhr) {{
                console.log('Loading: ' + (xhr.loaded / xhr.total * 100) + '%');
            }},
            function (error) {{
                console.error('Error loading GLTF:', error);
                document.getElementById('anim-info').innerHTML = 'Error loading model: ' + error;
            }}
        );
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            
            const delta = clock.getDelta();
            
            if (mixer) {{
                mixer.update(delta);
            }}
            
            controls.update();
            renderer.render(scene, camera);
        }}
        
        animate();
        
        // Handle window resize
        window.addEventListener('resize', function() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""
    return html

