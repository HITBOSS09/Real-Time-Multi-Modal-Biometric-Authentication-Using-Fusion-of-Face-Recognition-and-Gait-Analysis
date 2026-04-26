"""
Face Registration Module for NiceGUI Dashboard
Handles enrollment of face embeddings for persons in the system.
"""

import logging
import os
import aiohttp
from nicegui import ui
from pathlib import Path

logger = logging.getLogger(__name__)


async def face_register_page(person_id: int):
    """Face registration workflow page."""
    
    from db import DatabaseConnector
    db_connector = DatabaseConnector()
    
    person = db_connector.get_person_by_id(person_id)
    if not person:
        ui.navigate.to('/register')
        return
    person_name = person.get('name', f"Person {person_id}")
    
    uploaded_face_images = []
    
    def update_face_file_label(file_names: list[str]):
        if not file_names:
            return 'No files selected'
        return f"{len(file_names)} image(s) selected: {', '.join(file_names)}"
    
    with ui.header().classes('bg-slate-950 text-white shadow-xl border-b border-slate-800'):
        with ui.row().classes('w-full justify-between items-center px-6 py-4'):
            ui.label('Face Biometric Enrollment').classes('text-2xl font-bold')
            ui.button('← Back', on_click=lambda: ui.navigate.to('/register')).props('flat').classes('text-slate-300')

    with ui.column().classes('w-full p-6').style('background: linear-gradient(180deg, #050b17 0%, #071024 100%); max-width: 1000px; margin: 0 auto;'):
        ui.label(f'Enrolling Face Biometrics for: {person_name}').classes('text-xl font-bold text-white mb-2')
        ui.label('person_id: ' + str(person_id)).classes('text-sm text-slate-400 mb-4')
        ui.label('Step 2 of 4: Capture 4 face samples').classes('text-sm text-slate-300 mb-4')
        ui.progress(value=50).classes('w-full mb-6')
        
        with ui.card().classes('w-full'):
            ui.markdown("""
            ### Face Enrollment Process
            
            To enroll face biometrics:
            
            1. **Camera Setup**
               - Position person 1-2 meters from camera
               - Ensure good lighting (natural or bright indoor lighting)
               - Face should be clearly visible without obstructions
            
            2. **Capture Process**
               - System will capture 4 face samples at different angles
               - Sample 1: Face forward
               - Sample 2: Head tilted 15° left
               - Sample 3: Head tilted 15° right
               - Sample 4: Head tilted up 15°
            
            3. **Quality Requirements**
               - Minimum face size: 60 pixels
               - Eyes clearly visible
               - No extreme lighting (shadows on face)
               - No severe head rotation
            
            ---
            
            ### Using Mobile Device
            
            1. Open camera app on mobile device
            2. Use live preview to position face correctly
            3. Capture 4 clear front-facing photos
            4. Upload them using the form below
            
            ### Using Webcam (Linux/Mac)
            
            ```bash
            # Install required packages
            pip install opencv-python pillow
            
            # Run capture script
            python3 capture_face_samples.py --person-id {person_id}
            ```
            """)
            
            ui.separator()
            
            ui.label('Upload Face Samples').classes('text-lg font-bold my-4')
            face_files_label = ui.label('No files selected').classes('text-sm text-gray-600 mb-2')
            
            async def on_face_images_upload(event):
                uploaded_face_images.clear()
                uploaded_face_images.extend(event.files)
                file_names = [file.name for file in event.files]
                face_files_label.text = update_face_file_label(file_names)

            with ui.row().classes('gap-4 w-full'):
                ui.upload(
                    label='Select Face Images',
                    multiple=True,
                    max_files=20,
                    on_multi_upload=on_face_images_upload,
                    accept='image/*'
                ).classes('flex-1')
                
                async def submit_face_samples():
                    if not uploaded_face_images:
                        ui.notify('Please select one or more face images before submitting')
                        return

                    face_service_url = os.getenv('FACE_SERVICE_URL', 'http://localhost:8000').rstrip('/')
                    sample_data = []
                    for file in uploaded_face_images:
                        sample_data.append({
                            'name': file.name or 'face_sample.jpg',
                            'content_type': file.content_type or 'application/octet-stream',
                            'data': file.read(),
                        })

                    try:
                        async with aiohttp.ClientSession() as session:
                            form = aiohttp.FormData()
                            for sample in sample_data:
                                form.add_field(
                                    'files',
                                    sample['data'],
                                    filename=sample['name'],
                                    content_type=sample['content_type']
                                )
                            async with session.post(f'{face_service_url}/add_face_samples/{person_id}', data=form) as response:
                                if response.status == 201:
                                    payload = await response.json()
                                    ui.notify(
                                        f"Uploaded {payload.get('samples_added', len(sample_data))} face image(s). "
                                        f"Total for person: {payload.get('total_samples_for_person', 'unknown')}"
                                    )
                                    ui.timer(3.0, lambda: ui.navigate.to('/register'))
                                else:
                                    text = await response.text()
                                    ui.notify(f'Face service error {response.status}: {text}')
                    except Exception as e:
                        logger.exception('Face sample submission failed')
                        save_dir = Path(__file__).resolve().parent / 'uploads' / 'face' / str(person_id)
                        save_dir.mkdir(parents=True, exist_ok=True)
                        for sample in sample_data:
                            target_path = save_dir / sample['name']
                            target_path.write_bytes(sample['data'])
                        ui.notify(
                            f'Face service unavailable. Saved {len(sample_data)} image(s) locally to {save_dir}'
                        )
                        ui.timer(3.0, lambda: ui.navigate.to('/register'))
                
                ui.button('📤 Submit Samples', on_click=submit_face_samples).classes('flex-1 bg-green-600')
            
            ui.separator()
            
            ui.markdown("""
            ### Alternative: Direct Camera Integration
            
            For production deployment with live camera capture:
            
            ```python
            # This will be implemented in v1.1.0
            import cv2
            from insightface.app import FaceAnalysis
            
            def capture_face_samples():
                cap = cv2.VideoCapture(0)
                samples = []
                
                # Capture 4 samples with quality checks
                for i in range(4):
                    ret, frame = cap.read()
                    # Face detection and embedding extraction
                    samples.append(embedding)
                
                # Store in database
                db_connector.store_face_embeddings(person_id, samples)
            ```
            """)

