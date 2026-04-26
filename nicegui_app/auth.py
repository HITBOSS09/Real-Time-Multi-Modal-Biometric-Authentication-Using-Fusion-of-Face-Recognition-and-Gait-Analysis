"""Authentication module for NiceGUI dashboard.

Simple login/signup system with in-memory user storage.
In production, this would use a proper authentication service.
"""

import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, ValidationError, constr


class UserCredentials(BaseModel):
    email: EmailStr
    password: constr(min_length=6)


class User:
    """Simple user model."""
    
    def __init__(self, email: str, password_hash: str, created_at: datetime = None):
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at or datetime.now()


class AuthManager:
    """Manages user authentication."""
    
    def __init__(self):
        # In-memory user store (in production: use database)
        self.users: dict[str, User] = {}
        self._current_user: Optional[str] = None
        
        # Create default admin user
        self._create_default_user()
    
    def _create_default_user(self):
        """Create a default admin user for testing."""
        admin_email = "admin@surveillance.local"
        admin_password = "admin123"
        if admin_email not in self.users:
            password_hash = self._hash_password(admin_password)
            self.users[admin_email] = User(admin_email, password_hash)
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def signup(self, email: str, password: str) -> tuple[bool, str]:
        """Create a new user account.
        
        Returns: (success, message)
        """
        try:
            credentials = UserCredentials(email=email, password=password)
        except ValidationError as e:
            return False, ' '.join(err['msg'] for err in e.errors())

        if credentials.email in self.users:
            self._current_user = credentials.email
            return True, 'User already exists. Logged in with existing account.'

        try:
            password_hash = self._hash_password(credentials.password)
            self.users[credentials.email] = User(credentials.email, password_hash)
            return True, 'Account created successfully'
        except Exception as e:
            return False, f"Signup failed: {str(e)}"
    
    def login(self, email: str, password: str) -> tuple[bool, str]:
        """Authenticate user.
        
        Returns: (success, message)
        """
        try:
            credentials = UserCredentials(email=email, password=password)
        except ValidationError as e:
            return False, ' '.join(err['msg'] for err in e.errors())

        if credentials.email not in self.users:
            return False, 'User not found'

        user = self.users[credentials.email]
        password_hash = self._hash_password(credentials.password)

        if user.password_hash != password_hash:
            return False, 'Invalid password'

        self._current_user = credentials.email
        return True, f"Welcome, {credentials.email}"
    
    def logout(self):
        """Logout current user."""
        self._current_user = None
    
    def get_current_user(self) -> Optional[str]:
        """Get currently logged-in user email."""
        return self._current_user
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self._current_user is not None
