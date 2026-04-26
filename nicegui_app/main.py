"""
MULTI FACTOR BIOMETRIC SURVEILLANCE SYSTEM
Main entry point for the NiceGUI dashboard application.

Integrates Face Recognition, Gait Recognition, and Hybrid Fusion Engine
with a professional surveillance control room interface.
"""

import json
import logging
import os
import re
from datetime import datetime
from nicegui import ui, app as nicegui_app
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from auth import AuthManager
from db import DatabaseConnector
from system_control import ServiceManager
from face_register import face_register_page
from gait_register import gait_register_page

# Global instances
auth_manager = AuthManager()
db_connector = DatabaseConnector()
service_manager = ServiceManager()

# Enable dark theme for the dashboard and optionally inject Firebase scripts.
FIREBASE_CONFIG = {
    'apiKey': os.getenv('FIREBASE_API_KEY', ''),
    'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN', ''),
    'projectId': os.getenv('FIREBASE_PROJECT_ID', ''),
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET', ''),
    'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID', ''),
    'appId': os.getenv('FIREBASE_APP_ID', ''),
}
FIREBASE_ENABLED = all(FIREBASE_CONFIG.values())

# Page state
current_user = {'logged_in': False, 'email': None}
recent_alerts: list[dict] = []
activity_logs: list[dict] = []
seen_fusion_ids: set[int] = set()


def is_valid_email(email: str) -> bool:
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email))


# ============================================================================
# LOGIN PAGE
# ============================================================================

async def login_page():
    """Login screen."""
    input_base_classes = 'w-full mb-3 text-white placeholder:text-slate-400 bg-slate-950'
    input_error_classes = input_base_classes + ' border border-red-500'

    def set_input_valid(input_element):
        input_element.classes(input_base_classes)

    def set_input_invalid(input_element):
        input_element.classes(input_error_classes)

    def validate_email_field(e):
        value = (e.value or '').strip()
        if not value:
            email_error.text = 'Email is required'
            set_input_invalid(email_input)
            return False
        if not is_valid_email(value):
            email_error.text = 'Invalid email format'
            set_input_invalid(email_input)
            return False
        email_error.text = ''
        set_input_valid(email_input)
        return True

    def validate_password_field(e):
        value = e.value or ''
        if not value:
            password_error.text = 'Password required'
            set_input_invalid(password_input)
            return False
        if len(value) < 6:
            password_error.text = 'Minimum 6 characters'
            set_input_invalid(password_input)
            return False
        password_error.text = ''
        set_input_valid(password_input)
        return True

    async def google_login(e):
        if not FIREBASE_ENABLED:
            ui.notify('Google auth is not configured. Set Firebase env vars.', color='red')
            return
        result = await ui.run_javascript(
            'return window.runFirebaseGoogleSignIn ? window.runFirebaseGoogleSignIn() : {success:false, error: "firebase not ready"};'
        )
        if not result:
            ui.notify('Google sign-in failed. Try again.', color='red')
            return
        if result.get('success') and result.get('email'):
            email = result['email']
            if email not in auth_manager.users:
                auth_manager.signup(email, 'google_auth_demo')
            current_user['logged_in'] = True
            current_user['email'] = email
            ui.navigate.to('/dashboard')
        else:
            ui.notify(f"Google sign-in failed: {result.get('error', 'unknown')}", color='red')

    async def handle_login(e):
        auth_message.text = ''
        valid_email = validate_email_field(email_input)
        valid_password = validate_password_field(password_input)
        if not valid_email or not valid_password:
            return
        email = (email_input.value or '').strip()
        password = password_input.value or ''
        success, message = auth_manager.login(email, password)
        if success:
            current_user['logged_in'] = True
            current_user['email'] = email
            ui.navigate.to('/dashboard')
        else:
            auth_message.text = message
            auth_message.classes('text-red-400')
            password_input.value = ''

    async def handle_signup(e):
        auth_message.text = ''
        valid_email = validate_email_field(email_input)
        valid_password = validate_password_field(password_input)
        if not valid_email or not valid_password:
            return
        email = (email_input.value or '').strip()
        password = password_input.value or ''
        success, message = auth_manager.signup(email, password)
        if success:
            auth_message.text = 'Account created! Log in now.'
            auth_message.classes('text-green-400')
            email_input.value = ''
            password_input.value = ''
            set_input_valid(email_input)
            set_input_valid(password_input)
        else:
            auth_message.text = message
            auth_message.classes('text-red-400')

    with ui.column().classes('w-full h-screen items-center justify-center').style(
        'background: radial-gradient(circle at top, rgba(37, 99, 235, .15), transparent 30%), '
        'linear-gradient(180deg, #020617 0%, #050f21 100%);'
    ):
        with ui.card().classes('w-full max-w-md p-8 shadow-2xl rounded-3xl bg-slate-900 border border-slate-800'):
            ui.label('MULTI FACTOR BIOMETRIC').classes('text-xl font-bold text-center text-white mb-4')
            ui.label('Secure access to the surveillance dashboard').classes('text-center text-slate-400 mb-6')

            email_input = ui.input('Email').props('outlined').classes(input_base_classes)
            email_error = ui.label('').classes('text-red-400 text-xs mt-1 mb-3')

            password_input = ui.input('Password', password=True).props('outlined').classes(input_base_classes)
            password_error = ui.label('').classes('text-red-400 text-xs mt-1 mb-4')

            auth_message = ui.label('').classes('text-sm min-h-[1rem] mb-4 text-slate-200')

            email_input.on('change', validate_email_field)
            password_input.on('change', validate_password_field)

            ui.button('Login', on_click=handle_login).classes('w-full bg-blue-600 text-white rounded-xl py-3 mb-3')
            ui.label('OR').classes('text-center text-slate-300 mb-3')
            if FIREBASE_ENABLED:
                ui.button('Continue with Google', on_click=google_login).classes('w-full bg-white text-black rounded-xl py-3 mb-3')
            ui.button('Signup', on_click=handle_signup).classes('w-full bg-blue-400 text-white rounded-xl py-3')


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

async def dashboard_page():
    """Main surveillance dashboard."""

    if not current_user['logged_in']:
        ui.navigate.to('/login')
        return

    def format_timestamp(value):
        if isinstance(value, datetime):
            return value.strftime('%H:%M:%S')
        return str(value)

    with ui.header().classes('bg-slate-950 text-white shadow-xl border-b border-slate-800'):
        with ui.row().classes('w-full justify-between items-center px-6 py-4'):
            ui.label('CCTV CONTROL ROOM').classes('text-2xl font-bold tracking-wider')
            with ui.row().classes('items-center gap-4'):
                ui.label(f"Operator: {current_user['email']}").classes('text-slate-300')

                async def handle_logout():
                    auth_manager.logout()
                    current_user['logged_in'] = False
                    current_user['email'] = None
                    ui.navigate.to('/login')

                ui.button('About', on_click=lambda: ui.navigate.to('/about')).props('flat').classes('text-slate-300')
                ui.button('Logout', on_click=handle_logout).props('flat').classes('text-slate-300')

    with ui.column().classes('w-full min-h-screen px-6 py-6 gap-4').style(
        'background: linear-gradient(180deg, #020613 0%, #061025 100%);'
    ):

        with ui.row().classes('w-full gap-4 flex-wrap'):
            # SYSTEM CONTROL PANEL
            with ui.column().classes('flex-1 min-w-[320px] gap-4'):
                with ui.card().classes('w-full p-6 bg-slate-900 border border-slate-800 shadow-2xl rounded-3xl'):
                    ui.label('SYSTEM CONTROL').classes('text-xl font-semibold text-white mb-4')
                    with ui.row().classes('flex-wrap gap-3 mb-5'):
                        status_labels = {
                            'face': ui.label('Face Engine · STOPPED').classes('text-sm text-rose-400 font-semibold bg-slate-800 rounded-full px-3 py-2'),
                            'gait': ui.label('Gait Engine · STOPPED').classes('text-sm text-rose-400 font-semibold bg-slate-800 rounded-full px-3 py-2'),
                            'fusion': ui.label('Fusion Engine · STOPPED').classes('text-sm text-rose-400 font-semibold bg-slate-800 rounded-full px-3 py-2'),
                        }
                    async def update_status():
                        statuses = service_manager.get_all_status()
                        for service, label in status_labels.items():
                            status = statuses.get(service, 'unknown')
                            label.text = f"{service.capitalize()} Engine · {status.upper()}"
                            if status == 'running':
                                label.classes('text-emerald-400 font-semibold bg-slate-800 rounded-full px-3 py-2')
                            elif status == 'paused':
                                label.classes('text-amber-400 font-semibold bg-slate-800 rounded-full px-3 py-2')
                            else:
                                label.classes('text-rose-400 font-semibold bg-slate-800 rounded-full px-3 py-2')

                    async def start_system():
                        ui.notify('Starting system...', color='green')
                        service_manager.start_system()
                        await update_status()
                        ui.notify('System started', color='green')

                    async def pause_system():
                        service_manager.pause_system()
                        await update_status()
                        ui.notify('System paused', color='orange')

                    async def stop_system():
                        service_manager.stop_system()
                        await update_status()
                        ui.notify('System stopped', color='red')

                    ui.label('Control actions').classes('text-slate-400 text-xs uppercase tracking-[0.18em]')
                    ui.button('▶ Start System', on_click=start_system).classes(
                        'w-full bg-emerald-500 hover:bg-emerald-400 text-slate-950 rounded-2xl py-3 transition-transform hover:scale-105'
                    )
                    ui.button('⏸ Pause System', on_click=pause_system).classes(
                        'w-full bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-2xl py-3 transition-transform hover:scale-105'
                    )
                    ui.button('⏹ Stop System', on_click=stop_system).classes(
                        'w-full bg-rose-600 hover:bg-rose-500 text-white rounded-2xl py-3 transition-transform hover:scale-105'
                    )
                    ui.button('👤 Register Person', on_click=lambda: ui.navigate.to('/register')).classes(
                        'w-full bg-slate-700 hover:bg-slate-600 text-white rounded-2xl py-3 transition-transform hover:scale-105'
                    )

                with ui.card().classes('w-full p-6 bg-slate-900 border border-slate-800 shadow-2xl rounded-3xl'):
                    ui.label('COMMAND SUMMARY').classes('text-xl font-semibold text-white mb-4')
                    ui.label('Live control room interface with real-time feed orchestration.').classes('text-slate-400 text-sm')

            # CAMERA PANEL
            with ui.column().classes('flex-2 min-w-[600px] gap-4'):
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    with ui.card().classes('flex-1 min-w-[320px] p-0 bg-slate-950 border border-slate-800 shadow-2xl rounded-3xl overflow-hidden'):
                        with ui.row().classes('items-center justify-between px-4 py-3 bg-slate-900'):
                            ui.label('Face Recognition Camera').classes('text-lg font-semibold text-white')
                            ui.label('● LIVE').classes('text-emerald-400 font-semibold')
                        with ui.div().classes('relative bg-black h-72'):
                            ui.image('http://192.168.0.102:8080/video').classes('w-full h-full object-cover')
                            with ui.row().classes('absolute top-3 left-3 items-center gap-2'):
                                ui.label('LIVE').classes('text-xs font-semibold uppercase text-emerald-200 bg-slate-950/80 rounded-full px-2 py-1')
                            face_clock = ui.label('', classes='absolute top-3 right-3 text-xs text-slate-200 bg-slate-950/80 rounded-full px-2 py-1')
                        ui.label('Waiting for stream...').classes('text-slate-400 text-sm px-4 py-3')

                    with ui.card().classes('flex-1 min-w-[320px] p-0 bg-slate-950 border border-slate-800 shadow-2xl rounded-3xl overflow-hidden'):
                        with ui.row().classes('items-center justify-between px-4 py-3 bg-slate-900'):
                            ui.label('Gait Recognition Camera').classes('text-lg font-semibold text-white')
                            ui.label('● LIVE').classes('text-emerald-400 font-semibold')
                        with ui.div().classes('relative bg-black h-72'):
                            ui.image('http://192.168.0.104:8080/video').classes('w-full h-full object-cover')
                            with ui.row().classes('absolute top-3 left-3 items-center gap-2'):
                                ui.label('LIVE').classes('text-xs font-semibold uppercase text-emerald-200 bg-slate-950/80 rounded-full px-2 py-1')
                            gait_clock = ui.label('', classes='absolute top-3 right-3 text-xs text-slate-200 bg-slate-950/80 rounded-full px-2 py-1')
                        ui.label('Waiting for stream...').classes('text-slate-400 text-sm px-4 py-3')

                with ui.row().classes('w-full gap-4 flex-wrap'):
                    with ui.card().classes('flex-1 min-w-[320px] p-4 bg-slate-900 border border-slate-800 shadow-2xl rounded-3xl'):
                        ui.label('REAL-TIME FUSION RESULTS').classes('text-xl font-semibold text-white mb-4')
                        fusion_table = ui.table(
                            columns=[
                                {'name': 'identity', 'label': 'Final Identity', 'field': 'identity'},
                                {'name': 'face_identity', 'label': 'Face ID', 'field': 'face_identity'},
                                {'name': 'gait_identity', 'label': 'Gait ID', 'field': 'gait_identity'},
                                {'name': 'face_score', 'label': 'Face Score', 'field': 'face_score'},
                                {'name': 'gait_score', 'label': 'Gait Score', 'field': 'gait_score'},
                                {'name': 'fused_score', 'label': 'Fused Score', 'field': 'fused_score'},
                                {'name': 'timestamp', 'label': 'Timestamp', 'field': 'timestamp'},
                            ],
                            rows=[],
                            pagination=False,
                        ).classes('w-full text-slate-200')

                    with ui.card().classes('flex-1 min-w-[320px] p-4 bg-slate-900 border border-slate-800 shadow-2xl rounded-3xl'):
                        ui.label('EVENT LOG').classes('text-xl font-semibold text-white mb-4')
                        log_text = ui.label('', classes='text-slate-200 whitespace-pre-wrap').style(
                            'min-height: 264px; display: block; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;'
                        )

            # ALERTS + STATS PANEL
            with ui.column().classes('flex-1 min-w-[320px] gap-4'):
                with ui.card().classes('w-full p-6 bg-slate-900 border border-slate-800 shadow-2xl rounded-3xl'):
                    ui.label('SYSTEM STATS').classes('text-xl font-semibold text-white mb-4')
                    with ui.row().classes('gap-4 flex-wrap'):
                        with ui.card().classes('flex-1 min-w-[140px] p-4 bg-slate-950 border border-slate-800 rounded-3xl shadow-xl'):
                            stat_total = ui.label('0').classes('text-4xl font-bold text-white')
                            ui.label('Total Fusions').classes('text-slate-400 text-sm mt-2')
                        with ui.card().classes('flex-1 min-w-[140px] p-4 bg-slate-950 border border-slate-800 rounded-3xl shadow-xl'):
                            stat_unique = ui.label('0').classes('text-4xl font-bold text-white')
                            ui.label('Unique IDs').classes('text-slate-400 text-sm mt-2')
                        with ui.card().classes('flex-1 min-w-[140px] p-4 bg-slate-950 border border-slate-800 rounded-3xl shadow-xl'):
                            stat_avg = ui.label('—').classes('text-4xl font-bold text-white')
                            ui.label('Avg Score').classes('text-slate-400 text-sm mt-2')

                with ui.card().classes('w-full p-6 bg-slate-900 border border-slate-800 shadow-2xl rounded-3xl') as alert_card:
                    ui.label('ALERTS').classes('text-xl font-semibold text-white mb-4')
                    alert_text = ui.label('', classes='text-red-300 whitespace-pre-wrap').style('min-height: 220px; display: block;')
                    ui.label('Latest alarms displayed first.').classes('text-slate-400 text-sm mt-3')

        async def update_dashboard():
            statuses = service_manager.get_all_status()
            for service, label in status_labels.items():
                status = statuses.get(service, 'unknown')
                label.text = f"{service.capitalize()} Engine · {status.upper()}"
                if status == 'running':
                    label.classes('text-emerald-400 font-semibold bg-slate-800 rounded-full px-3 py-2')
                elif status == 'paused':
                    label.classes('text-amber-400 font-semibold bg-slate-800 rounded-full px-3 py-2')
                else:
                    label.classes('text-rose-400 font-semibold bg-slate-800 rounded-full px-3 py-2')

            stat_dict = db_connector.get_fusion_stats()
            stat_total.text = f"{stat_dict.get('total_fusions', 0)}"
            stat_unique.text = f"{stat_dict.get('unique_identities', 0)}"
            avg = stat_dict.get('avg_score')
            stat_avg.text = f"{avg:.3f}" if avg else '—'

            results = db_connector.get_fusion_results(limit=20)
            fusion_table.rows = [
                {**row, 'timestamp': format_timestamp(row.get('timestamp', '—'))}
                for row in results
            ]

            new_alerts = []
            for row in results:
                row_id = int(row.get('id', 0) or 0)
                if row_id and row_id not in seen_fusion_ids:
                    seen_fusion_ids.add(row_id)
                    if row.get('identity') == 'Unknown' or row.get('face_identity') == 'Unknown' or row.get('gait_identity') == 'Unknown':
                        new_alerts.append({
                            'timestamp': format_timestamp(row.get('timestamp', datetime.now())),
                            'message': '⚠ Unknown Person Detected',
                        })
                        ui.notify('⚠ Unknown Person Detected', color='red')
                    activity_logs.insert(0, {
                        'timestamp': format_timestamp(row.get('timestamp', datetime.now())),
                        'message': f"Fusion event: {row.get('identity', 'Unknown')}"
                    })

            if new_alerts:
                recent_alerts[0:0] = new_alerts
            alert_text.text = '\n'.join(
                f"[{item['timestamp']}] {item['message']}" for item in recent_alerts[:10]
            )
            if recent_alerts:
                alert_card.classes('w-full p-6 bg-slate-900 border border-rose-500 shadow-2xl rounded-3xl animate-pulse')
            else:
                alert_card.classes('w-full p-6 bg-slate-900 border border-slate-800 shadow-2xl rounded-3xl')

            if activity_logs:
                log_text.text = '\n'.join(
                    f"[{item['timestamp']}] {item['message']}" for item in activity_logs[:20]
                )

            face_clock.text = datetime.now().strftime('%H:%M:%S')
            gait_clock.text = datetime.now().strftime('%H:%M:%S')

        await update_dashboard()
        ui.timer(1.0, update_dashboard)


# ============================================================================
# REGISTRATION PAGE
# ============================================================================

async def register_page():
    """Person registration page."""

    if not current_user['logged_in']:
        ui.navigate.to('/login')
        return

    with ui.header().classes('bg-slate-950 text-white shadow-xl border-b border-slate-800'):
        with ui.row().classes('w-full justify-between items-center px-6 py-4'):
            ui.label('Person Enrollment Workflow').classes('text-2xl font-bold')
            ui.button('← Back to Dashboard', on_click=lambda: ui.navigate.to('/dashboard')).props('flat').classes('text-slate-300')

    with ui.column().classes('w-full px-6 py-6 gap-6').style('background: linear-gradient(180deg, #050b17 0%, #071024 100%);'):
        with ui.card().classes('w-full p-6 bg-slate-900 border border-slate-800 shadow-2xl rounded-3xl'):
            ui.label('Enrollment Progress').classes('text-xl font-semibold text-white')
            ui.label('Follow the guided flow to register a new user with both face and gait samples.').classes('text-slate-400 mb-4')
            ui.progress(value=25).classes('w-full mb-6')
            with ui.row().classes('flex-wrap gap-4'):
                for step_text, active in [
                    ('1. Create User', True),
                    ('2. Capture Face', False),
                    ('3. Capture Gait', False),
                    ('4. Save Identity', False),
                ]:
                    with ui.card().classes('flex-1 min-w-[180px] p-4 bg-slate-800 border border-slate-700 rounded-3xl'):
                        ui.label(step_text).classes('text-sm font-semibold text-white' if active else 'text-slate-300')

        with ui.row().classes('w-full gap-6 flex-wrap'):
            with ui.card().classes('flex-1 min-w-[320px] p-6 bg-slate-900 border border-slate-800 shadow-2xl rounded-3xl'):
                ui.label('Step 1: Create User').classes('text-xl font-semibold text-white mb-4')
                ui.label('Start by creating a new identity record in the system.').classes('text-slate-400 mb-4')
                person_name = ui.input('Person Name', placeholder='Unique identity name').props('outlined').classes('w-full')
                person_email = ui.input('Email', placeholder='name@company.com').props('outlined type=email').classes('w-full mt-4')
                register_error = ui.label('').classes('text-red-400 text-sm mt-2')

                async def create_user():
                    register_error.text = ''
                    if not person_name.value:
                        register_error.text = 'Please enter a person name.'
                        return
                    if person_email.value and not is_valid_email(person_email.value):
                        register_error.text = 'Email must be valid if provided.'
                        return

                    existing_person = db_connector.get_person_by_name(person_name.value)
                    if existing_person:
                        ui.notify(f"Person '{person_name.value}' already exists. Continuing enrollment.", color='blue')
                        ui.navigate.to(f'/face-register/{existing_person["id"]}')
                        return

                    person_id = db_connector.create_person(person_name.value)
                    if person_id:
                        ui.notify('Person created. Proceed to face capture.', color='green')
                        ui.navigate.to(f'/face-register/{person_id}')
                    else:
                        register_error.text = 'Unable to create person. Name may already exist.'

                ui.button('Create User', on_click=create_user).classes(
                    'w-full bg-sky-600 hover:bg-sky-500 text-white rounded-2xl py-3 transition-transform hover:scale-105 mt-4'
                )

            with ui.card().classes('flex-1 min-w-[320px] p-6 bg-slate-900 border border-slate-800 shadow-2xl rounded-3xl'):
                ui.label('Why this workflow?').classes('text-xl font-semibold text-white mb-4')
                ui.markdown(
                    '- Step 1 creates a new person record in the database.\n'
                    '- Step 2 captures 4 face samples for face recognition.\n'
                    '- Step 3 uploads gait video for gait model enrollment.\n'
                    '- Step 4 confirms the identity and returns to the dashboard.\n'
                ).classes('text-slate-400')
# ============================================================================
# ABOUT PAGE
# ============================================================================

async def about_page():
    """About page."""
    
    with ui.header().classes('bg-gray-900 text-white shadow-lg'):
        with ui.row().classes('w-full justify-between items-center px-6 py-4'):
            ui.label('About').classes('text-2xl font-bold')
            ui.button('← Back to Dashboard', on_click=lambda: ui.navigate.to('/dashboard')).props('flat')
    
    with ui.column().classes('w-full p-6').style('background: #f5f5f5; max-width: 800px; margin: 0 auto;'):
        with ui.card().classes('w-full'):
            ui.label('MULTI FACTOR BIOMETRIC SURVEILLANCE SYSTEM').classes('text-2xl font-bold mb-4')
            
            ui.markdown("""
            ### System Overview
            
            This is a professional biometric surveillance platform that integrates:
            
            - **Face Recognition Engine**: Real-time face detection and identification using InsightFace (RetinaFace)
            - **Gait Recognition Engine**: Walking pattern analysis for identity confirmation
            - **Hybrid Fusion Engine**: Multimodal decision fusion for robust identification
            
            ### Technical Stack
            
            - **Backend**: Python (InsightFace, OpenCV, PostgreSQL)
            - **Frontend**: NiceGUI (web-based dashboard)
            - **Databases**: PostgreSQL (face DB), SQLite (gait DB)
            - **Acceleration**: CUDA GPU support for real-time processing
            
            ### Features
            
            ✓ Live CCTV camera monitoring\n
            ✓ Real-time person identification\n
            ✓ Multi-modal fusion decision\n
            ✓ Person enrollment and management\n
            ✓ Historical recognition logs\n
            ✓ System performance monitoring\n
            
            ### Security & Compliance
            
            - All biometric data encrypted in transit
            - PostgreSQL with SSL connections
            - User authentication and authorization
            - Complete audit logging
            
            ### Contact & Support
            
            For technical support or deployment assistance, please contact the engineering team.
            """)


# ============================================================================
# SETUP ROUTES
# ============================================================================

@ui.page('/')
async def index():
    """Render the login page at root."""
    await login_page()


@ui.page('/login')
async def login():
    await login_page()


@ui.page('/dashboard')
async def dashboard():
    await dashboard_page()


@ui.page('/register')
async def register():
    await register_page()


@ui.page('/about')
async def about():
    await about_page()


@ui.page('/face-register/{person_id}')
async def face_register(person_id: int):
    await face_register_page(person_id)


@ui.page('/gait-register/{person_id}')
async def gait_register(person_id: int):
    await gait_register_page(person_id)


# ============================================================================
# MAIN
# ============================================================================

if __name__ in {'__main__', '__mp_main__'}:
    logger.info("Starting MULTI FACTOR BIOMETRIC SURVEILLANCE SYSTEM")

    # Run the application
    ui.run(
        title='MULTI FACTOR BIOMETRIC SURVEILLANCE SYSTEM',
        host='0.0.0.0',
        port=8050,
        show=True,
        reload=False,
    )
