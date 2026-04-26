"""System control module for managing biometric services.

Handles starting, pausing, and stopping face, gait, and fusion engines.
"""

import subprocess
import os
import signal
import time
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages biometric system services."""
    
    def __init__(self):
        self.processes: Dict[str, Optional[subprocess.Popen]] = {
            'face': None,
            'gait': None,
            'fusion': None,
        }
        self.status: Dict[str, str] = {
            'face': 'stopped',
            'gait': 'stopped',
            'fusion': 'stopped',
        }
        self.base_path = os.path.expanduser('~/Downloads/biometric_system')
    
    def start_system(self) -> Dict[str, bool]:
        """Start all services.
        
        Returns: {service: started_successfully}
        """
        results = {}
        
        # Start face recognition engine
        results['face'] = self._start_face_engine()
        time.sleep(1)
        
        # Start gait recognition engine
        results['gait'] = self._start_gait_engine()
        time.sleep(1)
        
        # Start fusion engine
        results['fusion'] = self._start_fusion_engine()
        
        return results
    
    def _start_face_engine(self) -> bool:
        """Start face recognition engine."""
        try:
            if self.processes['face'] is not None:
                logger.warning("Face engine already running")
                return True
            
            face_service_path = os.path.join(self.base_path, 'face_service')
            cmd = [
                'python', '-m', 'cctv_engine.video_processor',
                '--rtsp', 'http://192.168.0.102:8080/video',
                '--camera', 'cam1'
            ]
            
            self.processes['face'] = subprocess.Popen(
                cmd,
                cwd=face_service_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.status['face'] = 'running'
            logger.info("Face engine started")
            return True
        except Exception as e:
            logger.error(f"Failed to start face engine: {e}")
            self.status['face'] = 'error'
            return False
    
    def _start_gait_engine(self) -> bool:
        """Start gait recognition engine."""
        try:
            if self.processes['gait'] is not None:
                logger.warning("Gait engine already running")
                return True
            
            gait_service_path = os.path.join(self.base_path, 'gait_service')
            cmd = [
                'python', '-m', 'multi_gait_system.run_recognition',
                '--video', 'http://192.168.0.104:8080/video',
                '--checkpoint', 'experiments/supervised/checkpoints/fold_1_seed_42.pt',
                '--device', 'cuda'
            ]
            
            self.processes['gait'] = subprocess.Popen(
                cmd,
                cwd=gait_service_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.status['gait'] = 'running'
            logger.info("Gait engine started")
            return True
        except Exception as e:
            logger.error(f"Failed to start gait engine: {e}")
            self.status['gait'] = 'error'
            return False
    
    def _start_fusion_engine(self) -> bool:
        """Start fusion engine."""
        try:
            if self.processes['fusion'] is not None:
                logger.warning("Fusion engine already running")
                return True
            
            fusion_engine_path = os.path.join(self.base_path, 'fusion_engine')
            cmd = ['python', 'fusion_service.py']
            
            self.processes['fusion'] = subprocess.Popen(
                cmd,
                cwd=fusion_engine_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.status['fusion'] = 'running'
            logger.info("Fusion engine started")
            return True
        except Exception as e:
            logger.error(f"Failed to start fusion engine: {e}")
            self.status['fusion'] = 'error'
            return False
    
    def pause_system(self) -> bool:
        """Pause all services (SIGSTOP)."""
        try:
            for service, proc in self.processes.items():
                if proc is not None and proc.poll() is None:
                    os.kill(proc.pid, signal.SIGSTOP)
                    self.status[service] = 'paused'
            logger.info("System paused")
            return True
        except Exception as e:
            logger.error(f"Failed to pause system: {e}")
            return False
    
    def resume_system(self) -> bool:
        """Resume all paused services (SIGCONT)."""
        try:
            for service, proc in self.processes.items():
                if proc is not None and proc.poll() is None and self.status[service] == 'paused':
                    os.kill(proc.pid, signal.SIGCONT)
                    self.status[service] = 'running'
            logger.info("System resumed")
            return True
        except Exception as e:
            logger.error(f"Failed to resume system: {e}")
            return False
    
    def stop_system(self) -> bool:
        """Stop all services."""
        try:
            for service, proc in self.processes.items():
                if proc is not None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    self.processes[service] = None
                    self.status[service] = 'stopped'
            logger.info("System stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
            return False
    
    def get_status(self, service: str) -> str:
        """Get status of a service."""
        return self.status.get(service, 'unknown')
    
    def get_all_status(self) -> Dict[str, str]:
        """Get status of all services."""
        return self.status.copy()
    
    def is_running(self, service: str) -> bool:
        """Check if a service is running."""
        proc = self.processes.get(service)
        if proc is None:
            return False
        return proc.poll() is None
