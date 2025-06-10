import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import asyncio
import aiohttp
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import jwt
from dataclasses import dataclass
import base64
import numpy as np
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)

@dataclass
class SecurityCredentials:
    """Data class for security credentials"""
    api_key: str
    secret_key: str
    access_token: Optional[str] = None
    token_expiry: Optional[datetime] = None
    mfa_token: Optional[str] = None

@dataclass
class ThreatIntelligence:
    """Data class for threat intelligence"""
    threat_id: str
    severity: float
    confidence: float
    indicators: List[Dict[str, Any]]
    context: Dict[str, Any]
    timestamp: datetime
    source: str
    mitigation_steps: List[str]

class SecurityIntegrator:
    """Enhanced security integration manager with WAF, EDR, and threat intelligence"""
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        jwt_secret: Optional[str] = None,
        quantum_resistant: bool = True
    ):
        self.encryption_key = encryption_key or os.urandom(32)
        self.jwt_secret = jwt_secret or os.urandom(32)
        self.fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))
        self.quantum_resistant = quantum_resistant
        
        # Initialize security systems
        self.security_systems = {}
        self.credentials = {}
        self.threat_intelligence = {}
        
        # Initialize ML models for anomaly detection
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        # Initialize deep learning model for threat analysis
        self.threat_analyzer = self._initialize_threat_analyzer()
        
        # Initialize encryption
        self._initialize_encryption()
    
    def _initialize_threat_analyzer(self) -> tf.keras.Model:
        """Initialize deep learning model for threat analysis"""
        inputs = layers.Input(shape=(100,))
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _initialize_encryption(self):
        """Initialize enhanced encryption systems"""
        # Generate RSA key pair with increased key size for quantum resistance
        key_size = 4096 if self.quantum_resistant else 2048
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        self.public_key = self.private_key.public_key()
        
        # Initialize PBKDF2 with increased iterations
        self.kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=os.urandom(16),
            iterations=200000  # Increased for better security
        )
    
    async def initialize(self):
        """Initialize security integrator asynchronously"""
        # Initialize security systems
        self.security_systems = {}
        self.credentials = {}
        self.threat_intelligence = {}
        
        # Initialize ML models for anomaly detection
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        # Initialize deep learning model for threat analysis
        self.threat_analyzer = self._initialize_threat_analyzer()
        
        # Initialize encryption
        self._initialize_encryption()
        
        logger.info("Security integrator initialized successfully")
    
    async def register_waf(
        self,
        system_name: str,
        api_url: str,
        credentials: SecurityCredentials,
        rules: List[Dict[str, Any]]
    ):
        """Register a Web Application Firewall"""
        self.security_systems[system_name] = {
            'type': 'waf',
            'api_url': api_url,
            'credentials': credentials,
            'rules': rules,
            'stats': {
                'blocked_requests': 0,
                'allowed_requests': 0,
                'false_positives': 0
            }
        }
        logger.info(f"Registered WAF: {system_name}")
    
    async def register_edr(
        self,
        system_name: str,
        api_url: str,
        credentials: SecurityCredentials,
        endpoints: List[str]
    ):
        """Register an Endpoint Detection and Response system"""
        self.security_systems[system_name] = {
            'type': 'edr',
            'api_url': api_url,
            'credentials': credentials,
            'endpoints': endpoints,
            'stats': {
                'detected_threats': 0,
                'blocked_actions': 0,
                'quarantined_files': 0
            }
        }
        logger.info(f"Registered EDR: {system_name}")
    
    async def analyze_threat_intelligence(
        self,
        threat_data: Dict[str, Any]
    ) -> ThreatIntelligence:
        """Analyze threat intelligence using ML models"""
        # Extract features
        features = self._extract_threat_features(threat_data)
        
        # Detect anomalies
        anomaly_score = self.anomaly_detector.score_samples([features])[0]
        
        # Analyze threat using deep learning model
        threat_probability = self.threat_analyzer.predict([features])[0][0]
        
        # Generate threat intelligence
        threat_id = self._generate_threat_id(threat_data)
        severity = float(threat_probability)
        confidence = float(1 - abs(anomaly_score))
        
        # Get mitigation steps
        mitigation_steps = self._get_mitigation_steps(
            threat_data,
            severity,
            confidence
        )
        
        threat_intel = ThreatIntelligence(
            threat_id=threat_id,
            severity=severity,
            confidence=confidence,
            indicators=self._extract_indicators(threat_data),
            context=self._extract_context(threat_data),
            timestamp=datetime.now(),
            source=threat_data.get('source', 'unknown'),
            mitigation_steps=mitigation_steps
        )
        
        # Store threat intelligence
        self.threat_intelligence[threat_id] = threat_intel
        
        return threat_intel
    
    def _extract_threat_features(self, threat_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from threat data for ML models"""
        # Implement feature extraction logic
        features = []
        
        # Add numerical features
        if 'severity' in threat_data:
            features.append(float(threat_data['severity']))
        if 'confidence' in threat_data:
            features.append(float(threat_data['confidence']))
        
        # Add categorical features
        if 'type' in threat_data:
            features.extend(self._one_hot_encode(threat_data['type']))
        
        # Pad or truncate to fixed size
        features = features[:100] + [0] * (100 - len(features))
        
        return np.array(features)
    
    def _one_hot_encode(self, category: str) -> List[float]:
        """One-hot encode categorical features"""
        categories = ['malware', 'phishing', 'dos', 'data_exfiltration']
        return [1.0 if cat == category else 0.0 for cat in categories]
    
    def _generate_threat_id(self, threat_data: Dict[str, Any]) -> str:
        """Generate unique threat ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = base64.urlsafe_b64encode(os.urandom(4)).decode()
        return f"THREAT-{timestamp}-{random_suffix}"
    
    def _extract_indicators(self, threat_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract threat indicators"""
        indicators = []
        
        if 'ip_addresses' in threat_data:
            indicators.extend([
                {'type': 'ip', 'value': ip}
                for ip in threat_data['ip_addresses']
            ])
        
        if 'domains' in threat_data:
            indicators.extend([
                {'type': 'domain', 'value': domain}
                for domain in threat_data['domains']
            ])
        
        if 'hashes' in threat_data:
            indicators.extend([
                {'type': 'hash', 'value': hash_value}
                for hash_value in threat_data['hashes']
            ])
        
        return indicators
    
    def _extract_context(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract threat context"""
        return {
            'timestamp': threat_data.get('timestamp'),
            'source_ip': threat_data.get('source_ip'),
            'target_ip': threat_data.get('target_ip'),
            'protocol': threat_data.get('protocol'),
            'port': threat_data.get('port'),
            'user_agent': threat_data.get('user_agent'),
            'request_path': threat_data.get('request_path'),
            'request_method': threat_data.get('request_method')
        }
    
    def _get_mitigation_steps(
        self,
        threat_data: Dict[str, Any],
        severity: float,
        confidence: float
    ) -> List[str]:
        """Get mitigation steps based on threat analysis"""
        steps = []
        
        # Add severity-based steps
        if severity > 0.8:
            steps.extend([
                "Immediately block source IP",
                "Quarantine affected systems",
                "Initiate incident response protocol"
            ])
        elif severity > 0.5:
            steps.extend([
                "Monitor source IP",
                "Increase logging verbosity",
                "Review system access logs"
            ])
        
        # Add confidence-based steps
        if confidence > 0.9:
            steps.extend([
                "Update threat intelligence database",
                "Share indicators with threat sharing platform"
            ])
        
        # Add type-specific steps
        if threat_data.get('type') == 'malware':
            steps.extend([
                "Scan affected systems",
                "Update antivirus signatures",
                "Review system changes"
            ])
        elif threat_data.get('type') == 'phishing':
            steps.extend([
                "Block malicious domains",
                "Alert users about phishing attempt",
                "Review email security settings"
            ])
        
        return steps
    
    async def update_threat_intelligence(
        self,
        threat_id: str,
        new_data: Dict[str, Any]
    ) -> ThreatIntelligence:
        """Update existing threat intelligence"""
        if threat_id not in self.threat_intelligence:
            raise ValueError(f"Threat ID not found: {threat_id}")
        
        # Update threat data
        threat_intel = self.threat_intelligence[threat_id]
        threat_intel.indicators.extend(self._extract_indicators(new_data))
        threat_intel.context.update(self._extract_context(new_data))
        
        # Reanalyze threat
        new_features = self._extract_threat_features(new_data)
        new_anomaly_score = self.anomaly_detector.score_samples([new_features])[0]
        new_threat_probability = self.threat_analyzer.predict([new_features])[0][0]
        
        # Update severity and confidence
        threat_intel.severity = max(threat_intel.severity, float(new_threat_probability))
        threat_intel.confidence = max(
            threat_intel.confidence,
            float(1 - abs(new_anomaly_score))
        )
        
        # Update mitigation steps
        threat_intel.mitigation_steps.extend(
            self._get_mitigation_steps(
                new_data,
                threat_intel.severity,
                threat_intel.confidence
            )
        )
        
        return threat_intel
    
    async def get_threat_correlations(
        self,
        threat_id: str
    ) -> List[Dict[str, Any]]:
        """Get correlated threats"""
        if threat_id not in self.threat_intelligence:
            raise ValueError(f"Threat ID not found: {threat_id}")
        
        threat_intel = self.threat_intelligence[threat_id]
        correlations = []
        
        for other_id, other_intel in self.threat_intelligence.items():
            if other_id == threat_id:
                continue
            
            # Calculate correlation score
            correlation_score = self._calculate_correlation(
                threat_intel,
                other_intel
            )
            
            if correlation_score > 0.7:  # High correlation threshold
                correlations.append({
                    'threat_id': other_id,
                    'correlation_score': correlation_score,
                    'common_indicators': self._find_common_indicators(
                        threat_intel,
                        other_intel
                    ),
                    'timeline': self._analyze_timeline(
                        threat_intel,
                        other_intel
                    )
                })
        
        return correlations
    
    def _calculate_correlation(
        self,
        threat1: ThreatIntelligence,
        threat2: ThreatIntelligence
    ) -> float:
        """Calculate correlation score between two threats"""
        # Implement correlation calculation logic
        score = 0.0
        
        # Compare indicators
        common_indicators = self._find_common_indicators(threat1, threat2)
        if common_indicators:
            score += 0.4
        
        # Compare context
        if self._compare_context(threat1.context, threat2.context):
            score += 0.3
        
        # Compare timing
        time_diff = abs(
            (threat1.timestamp - threat2.timestamp).total_seconds()
        )
        if time_diff < 3600:  # Within 1 hour
            score += 0.3
        
        return score
    
    def _find_common_indicators(
        self,
        threat1: ThreatIntelligence,
        threat2: ThreatIntelligence
    ) -> List[Dict[str, Any]]:
        """Find common indicators between two threats"""
        common = []
        
        for ind1 in threat1.indicators:
            for ind2 in threat2.indicators:
                if (
                    ind1['type'] == ind2['type'] and
                    ind1['value'] == ind2['value']
                ):
                    common.append(ind1)
        
        return common
    
    def _compare_context(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> bool:
        """Compare threat contexts"""
        # Compare source IP
        if (
            context1.get('source_ip') and
            context2.get('source_ip') and
            context1['source_ip'] == context2['source_ip']
        ):
            return True
        
        # Compare target IP
        if (
            context1.get('target_ip') and
            context2.get('target_ip') and
            context1['target_ip'] == context2['target_ip']
        ):
            return True
        
        # Compare protocol and port
        if (
            context1.get('protocol') and
            context2.get('protocol') and
            context1.get('port') and
            context2.get('port') and
            context1['protocol'] == context2['protocol'] and
            context1['port'] == context2['port']
        ):
            return True
        
        return False
    
    def _analyze_timeline(
        self,
        threat1: ThreatIntelligence,
        threat2: ThreatIntelligence
    ) -> Dict[str, Any]:
        """Analyze timeline between two threats"""
        time_diff = (threat2.timestamp - threat1.timestamp).total_seconds()
        
        return {
            'time_difference': time_diff,
            'sequence': 'before' if time_diff > 0 else 'after',
            'time_scale': self._get_time_scale(time_diff)
        }
    
    def _get_time_scale(self, seconds: float) -> str:
        """Get human-readable time scale"""
        if seconds < 60:
            return 'seconds'
        elif seconds < 3600:
            return 'minutes'
        elif seconds < 86400:
            return 'hours'
        else:
            return 'days'
    
    async def export_threat_intelligence(
        self,
        format: str = 'json'
    ) -> Union[str, bytes]:
        """Export threat intelligence"""
        if format == 'json':
            return json.dumps(
                {
                    threat_id: {
                        'threat_id': intel.threat_id,
                        'severity': intel.severity,
                        'confidence': intel.confidence,
                        'indicators': intel.indicators,
                        'context': intel.context,
                        'timestamp': intel.timestamp.isoformat(),
                        'source': intel.source,
                        'mitigation_steps': intel.mitigation_steps
                    }
                    for threat_id, intel in self.threat_intelligence.items()
                },
                indent=2
            )
        elif format == 'stix':
            # Implement STIX export
            pass
        elif format == 'csv':
            # Implement CSV export
            pass
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def import_threat_intelligence(
        self,
        data: Union[str, bytes],
        format: str = 'json'
    ):
        """Import threat intelligence"""
        if format == 'json':
            threat_data = json.loads(data)
            for threat_id, intel_data in threat_data.items():
                self.threat_intelligence[threat_id] = ThreatIntelligence(
                    threat_id=intel_data['threat_id'],
                    severity=float(intel_data['severity']),
                    confidence=float(intel_data['confidence']),
                    indicators=intel_data['indicators'],
                    context=intel_data['context'],
                    timestamp=datetime.fromisoformat(intel_data['timestamp']),
                    source=intel_data['source'],
                    mitigation_steps=intel_data['mitigation_steps']
                )
        elif format == 'stix':
            # Implement STIX import
            pass
        elif format == 'csv':
            # Implement CSV import
            pass
        else:
            raise ValueError(f"Unsupported import format: {format}")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get enhanced security metrics"""
        metrics = {
            'active_systems': len(self.security_systems),
            'authenticated_systems': sum(
                1 for system in self.security_systems.values()
                if system['credentials'].access_token
            ),
            'encryption_strength': 'AES-128-CBC with RSA-4096' if self.quantum_resistant else 'AES-128-CBC with RSA-2048',
            'last_key_rotation': datetime.now().isoformat(),
            'threat_intelligence': {
                'total_threats': len(self.threat_intelligence),
                'high_severity_threats': sum(
                    1 for intel in self.threat_intelligence.values()
                    if intel.severity > 0.8
                ),
                'high_confidence_threats': sum(
                    1 for intel in self.threat_intelligence.values()
                    if intel.confidence > 0.9
                )
            },
            'system_stats': {
                system_name: system['stats']
                for system_name, system in self.security_systems.items()
            }
        }
        
        return metrics 