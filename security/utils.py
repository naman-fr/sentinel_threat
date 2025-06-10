import os
import yaml
import logging
import hashlib
import secrets
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class SecurityUtils:
    """Utility class for security operations"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load security configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load security config: {e}")
            raise
    
    @staticmethod
    def generate_secure_password(length: int = 32) -> str:
        """Generate a secure random password"""
        alphabet = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "!@#$%^&*()_+-=[]{}|;:,.<>?"
        )
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
        """Hash a password using PBKDF2"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        
        key = kdf.derive(password.encode())
        return key, salt
    
    @staticmethod
    def verify_password(password: str, stored_key: bytes, salt: bytes) -> bool:
        """Verify a password against stored hash"""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000
            )
            
            kdf.verify(password.encode(), stored_key)
            return True
        except Exception:
            return False
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key"""
        return base64.urlsafe_b64encode(os.urandom(32)).decode()
    
    @staticmethod
    def generate_jwt_token(
        payload: Dict,
        secret: str,
        expires_in: int = 3600
    ) -> str:
        """Generate a JWT token"""
        payload['exp'] = datetime.utcnow() + timedelta(seconds=expires_in)
        return jwt.encode(payload, secret, algorithm='HS256')
    
    @staticmethod
    def verify_jwt_token(token: str, secret: str) -> Dict:
        """Verify a JWT token"""
        try:
            return jwt.decode(token, secret, algorithms=['HS256'])
        except jwt.InvalidTokenError as e:
            logger.error(f"JWT verification failed: {e}")
            raise
    
    @staticmethod
    def encrypt_file(
        file_path: str,
        key: bytes
    ) -> tuple[bytes, bytes]:
        """Encrypt a file using Fernet"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            f = Fernet(key)
            encrypted_data = f.encrypt(data)
            return encrypted_data, hashlib.sha256(data).digest()
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            raise
    
    @staticmethod
    def decrypt_file(
        encrypted_data: bytes,
        key: bytes,
        output_path: str
    ) -> bool:
        """Decrypt a file using Fernet"""
        try:
            f = Fernet(key)
            decrypted_data = f.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            return True
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            return False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize a filename to prevent path traversal"""
        # Remove directory components
        filename = os.path.basename(filename)
        
        # Remove null bytes
        filename = filename.replace('\0', '')
        
        # Remove potentially dangerous characters
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        return filename
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate an IP address"""
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            
            return all(
                0 <= int(part) <= 255
                for part in parts
            )
        except (AttributeError, TypeError, ValueError):
            return False
    
    @staticmethod
    def validate_domain(domain: str) -> bool:
        """Validate a domain name"""
        try:
            # Basic domain validation
            if not domain or len(domain) > 255:
                return False
            
            # Check for valid characters
            allowed = set('abcdefghijklmnopqrstuvwxyz'
                         'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                         '0123456789-._')
            
            return all(c in allowed for c in domain)
        except (AttributeError, TypeError):
            return False
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        try:
            sha256_hash = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                for byte_block in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(byte_block)
            
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"File hash calculation failed: {e}")
            raise
    
    @staticmethod
    def validate_certificate(cert_path: str) -> bool:
        """Validate a certificate file"""
        try:
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
            
            # Basic certificate validation
            if not cert_data.startswith(b'-----BEGIN CERTIFICATE-----'):
                return False
            
            if not cert_data.endswith(b'-----END CERTIFICATE-----\n'):
                return False
            
            return True
        except Exception as e:
            logger.error(f"Certificate validation failed: {e}")
            return False
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def mask_sensitive_data(data: str, mask_char: str = '*') -> str:
        """Mask sensitive data in a string"""
        # Common patterns for sensitive data
        patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),
            (r'\b\d{16}\b', 'credit_card'),
            (r'\b\d{3}-\d{2}-\d{4}\b', 'ssn'),
            (r'\b\d{10}\b', 'phone')
        ]
        
        import re
        for pattern, data_type in patterns:
            if data_type == 'email':
                # Keep first and last character of local part
                data = re.sub(
                    pattern,
                    lambda m: m.group()[0] + mask_char * (len(m.group().split('@')[0]) - 2) + m.group()[-1] + '@' + m.group().split('@')[1],
                    data
                )
            else:
                # Mask all but last 4 digits
                data = re.sub(
                    pattern,
                    lambda m: mask_char * (len(m.group()) - 4) + m.group()[-4:],
                    data
                )
        
        return data 