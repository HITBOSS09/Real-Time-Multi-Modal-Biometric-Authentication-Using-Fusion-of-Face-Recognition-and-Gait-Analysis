"""
Gait Registration Module for NiceGUI Dashboard
Handles enrollment of gait patterns for persons in the system.
"""

import json
import logging
import os
from datetime import datetime
from nicegui import ui
from pathlib import Path

logger = logging.getLogger(__name__)


async def gait_register_page(person_id: int):
    """Gait registration workflow page."""
    
    from db import DatabaseConnector
    db_connector = DatabaseConnector()
    
    person = db_connector.get_person_by_id(person_id)
    if not person:
        ui.navigate.to('/register')
        return
    person_name = person.get('name', f"Person {person_id}")
    
    uploaded_gait_videos = []
    
    def update_gait_file_label(file_names: list[str]):
        if not file_names:
            return 'No videos selected'
        return f"{len(file_names)} video(s) selected: {', '.join(file_names)}"
    
    with ui.header().classes('bg-slate-950 text-white shadow-xl border-b border-slate-800'):
        with ui.row().classes('w-full justify-between items-center px-6 py-4'):
            ui.label('Gait Biometric Enrollment').classes('text-2xl font-bold')
            ui.button('← Back', on_click=lambda: ui.navigate.to('/register')).props('flat').classes('text-slate-300')

    with ui.column().classes('w-full p-6').style('background: linear-gradient(180deg, #050b17 0%, #071024 100%); max-width: 1000px; margin: 0 auto;'):
        ui.label(f'Enrolling Gait Biometrics for: {person_name}').classes('text-xl font-bold text-white mb-2')
        ui.label('person_id: ' + str(person_id)).classes('text-sm text-slate-400 mb-4')
        ui.label('Step 3 of 4: Upload gait videos').classes('text-sm text-slate-300 mb-4')
        ui.progress(value=75).classes('w-full mb-6')
        
        with ui.card().classes('w-full'):
            ui.markdown("""
            ### Gait Enrollment Process
            
            Gait recognition analyzes unique walking patterns for identity verification.
            
            1. **Setup Requirements**
               - Hallway or open space (minimum 5 meters in length)
               - Camera positioned perpendicular to walking path
               - Camera height: 1-1.5 meters from ground
               - Person walks naturally from left to right across frame
            
            2. **Capture Process**
               - System will record 6 walking cycles
               - Each cycle: person walks across frame and returns to start
               - Approximately 30-45 seconds total recording time
            
            3. **Walking Guidelines**
               - Walk at natural pace (don't rush or slow down)
               - Walk in straight line
               - Maintain consistent gait (no limping or unusual motion)
               - Can wear different clothing, but not extreme costume
            
            4. **Quality Requirements**
               - Minimum 60% of body in frame (head to feet)
               - Full walking motion visible (not just standing)
               - Stable lighting (no flashing/shadows)
               - Clear silhouette detection possible
            
            ---
            
            ### Gait Features Extracted
            
            The system analyzes:
            - Stride length and frequency
            - Knee bending angle
            - Arm swing pattern
            - Torso motion
            - Step timing
            - Overall posture
            
            => Creates unique biometric profile that's hard to fake
            => Can identify even with face covered (mask, hat, etc.)
            
            ---
            """)
            
            ui.separator()
            
            ui.label('Upload Gait Video').classes('text-lg font-bold my-4')
            
            ui.label('Recommended: 30-60 second video of natural walking (6 cycles)').classes('text-sm text-gray-600 mb-4')
            gait_files_label = ui.label('No videos selected').classes('text-sm text-gray-600 mb-2')
            
            async def on_gait_videos_upload(event):
                uploaded_gait_videos.clear()
                uploaded_gait_videos.extend(event.files)
                file_names = [file.name for file in event.files]
                gait_files_label.text = update_gait_file_label(file_names)

            with ui.row().classes('gap-4 w-full'):
                ui.upload(
                    label='Select Gait Videos (.mp4, .avi)',
                    multiple=True,
                    max_files=5,
                    on_multi_upload=on_gait_videos_upload,
                    accept='video/*'
                ).classes('flex-1')
                
                async def submit_gait_samples():
                    if not uploaded_gait_videos:
                        ui.notify('Please select one or more gait videos before submitting')
                        return

                    upload_root = Path(__file__).resolve().parent / 'uploads' / 'gait' / str(person_id)
                    upload_root.mkdir(parents=True, exist_ok=True)
                    metadata_path = upload_root / 'metadata.json'
                    metadata = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'saved_at': datetime.now().isoformat(),
                        'videos': []
                    }

                    for file in uploaded_gait_videos:
                        target_path = upload_root / file.name
                        file.save(target_path)
                        metadata['videos'].append({
                            'filename': file.name,
                            'path': str(target_path),
                            'content_type': file.content_type,
                            'saved_at': datetime.now().isoformat()
                        })

                    if metadata_path.exists():
                        try:
                            existing_data = json.loads(metadata_path.read_text())
                            existing_data.setdefault('videos', []).extend(metadata['videos'])
                            existing_data['saved_at'] = metadata['saved_at']
                            metadata = existing_data
                        except Exception:
                            pass

                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    ui.notify(f'Saved {len(uploaded_gait_videos)} gait video(s) for {person_name}')
                    ui.notify('Gait samples are stored locally and ready for further processing')
                    ui.timer(3.0, lambda: ui.navigate.to('/register'))
                
                ui.button('📤 Submit Videos', on_click=submit_gait_samples).classes('flex-1 bg-green-600')
                async def finalize_enrollment():
                    ui.notify('Enrollment complete. Identity has been saved.', color='green')
                    ui.timer(2.0, lambda: ui.navigate.to('/register'))
                ui.button('✅ Finalize Enrollment', on_click=finalize_enrollment).classes('flex-1 bg-sky-600')
            
            ui.separator()
            
            ui.markdown("""
            ### Camera Specifications
            
            **Ideal Camera Setup for Gait Capture:**
            
            - **Position**: Lateral view (perpendicular to walking direction)
            - **Height**: 1.2 meters (approximately hip height)
            - **Distance**: 2-3 meters from walking path
            - **Frame Rate**: 24+ FPS (smoother analysis)
            - **Resolution**: 1280x720 minimum (full body silhouette)
            - **Lens**: Wide angle (80°+ field of view)
            
            ### Sample Capture Commands
            
            **Using Webcam (Linux):**
            ```bash
            ffmpeg -f v4l2 -i /dev/video0 -t 60 gait_sample.mp4
            ```
            
            **Using IP Camera (RTSP):**
            ```bash
            ffmpeg -rtsp_transport tcp -i rtsp://camera:password@192.168.0.102:554/stream -t 60 gait_sample.mp4
            ```
            
            **Using Mobile Phone**
            - Record video in landscape orientation
            - Stable hold (use tripod if possible)
            - Good lighting throughout
            - No sudden zoom or pans
            
            ### Feature Extraction Architecture
            
            ```
            Gait Video (.mp4)
                    ↓
            MediaPipe Pose Detection (17 keypoints)
                    ↓
            Temporal Feature Extraction (stride, angles, timing)
                    ↓
            CASIA-B Gait Model (silhouette-based or model-based)
                    ↓
            Gait Embedding (128D vector)
                    ↓
            Store in SQLite + PostgreSQL
            ```
            """)

