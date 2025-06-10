import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class BehaviorPattern:
    """Data class for behavior patterns"""
    pattern_id: str
    timestamp: datetime
    features: np.ndarray
    confidence: float
    anomaly_score: float
    context: Dict

class BehavioralLSTM(nn.Module):
    """LSTM-based behavioral analysis model"""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with attention mechanism
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and attention weights
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, attn_weights = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)
        
        # Classification
        behavior_score = self.classifier(attn_out.mean(dim=1))
        anomaly_score = self.anomaly_detector(attn_out.mean(dim=1))
        
        outputs = {
            'behavior_score': behavior_score,
            'anomaly_score': anomaly_score,
            'features': attn_out
        }
        
        if return_attention:
            outputs['attention_weights'] = attn_weights
            
        return outputs

class BehavioralAnalyzer:
    """High-level behavioral analysis interface"""
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = BehavioralLSTM(
            input_size=128,  # Adjust based on your feature size
            hidden_size=256
        ).to(device)
        
        if model_path:
            self.load_model(model_path)
        
        # Initialize pattern database
        self.pattern_database = []
        self.anomaly_threshold = 0.8

    async def initialize(self):
        """Initialize behavioral analyzer asynchronously"""
        # Initialize model
        self.model = BehavioralLSTM(
            input_size=128,  # Adjust based on your feature size
            hidden_size=256
        ).to(self.device)
        
        # Initialize pattern database
        self.pattern_database = []
        self.anomaly_threshold = 0.8
        
        logger.info("Behavioral analyzer initialized successfully")
    
    def load_model(self, path: str):
        """Load trained model weights"""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def analyze_behavior(
        self,
        features: np.ndarray,
        context: Dict
    ) -> BehaviorPattern:
        """
        Analyze behavior patterns
        
        Args:
            features: Behavior features array
            context: Additional context information
            
        Returns:
            BehaviorPattern object containing analysis results
        """
        # Preprocess features
        features_tensor = self._preprocess_features(features)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(features_tensor)
        
        # Create behavior pattern
        pattern = BehaviorPattern(
            pattern_id=self._generate_pattern_id(),
            timestamp=datetime.now(),
            features=features,
            confidence=outputs['behavior_score'].item(),
            anomaly_score=outputs['anomaly_score'].item(),
            context=context
        )
        
        # Update pattern database
        self._update_pattern_database(pattern)
        
        return pattern
    
    def detect_anomalies(
        self,
        patterns: List[BehaviorPattern],
        window_size: int = 10
    ) -> List[BehaviorPattern]:
        """
        Detect anomalous behavior patterns
        
        Args:
            patterns: List of behavior patterns
            window_size: Size of sliding window for analysis
            
        Returns:
            List of anomalous patterns
        """
        anomalous_patterns = []
        
        for i in range(len(patterns) - window_size + 1):
            window = patterns[i:i + window_size]
            
            # Calculate window statistics
            anomaly_scores = [p.anomaly_score for p in window]
            mean_score = np.mean(anomaly_scores)
            std_score = np.std(anomaly_scores)
            
            # Detect anomalies
            for pattern in window:
                z_score = (pattern.anomaly_score - mean_score) / std_score
                if z_score > 2.0:  # More than 2 standard deviations
                    anomalous_patterns.append(pattern)
        
        return anomalous_patterns
    
    def predict_behavior(
        self,
        current_pattern: BehaviorPattern,
        horizon: int = 5
    ) -> List[Dict]:
        """
        Predict future behavior patterns
        
        Args:
            current_pattern: Current behavior pattern
            horizon: Number of future steps to predict
            
        Returns:
            List of predicted behavior patterns
        """
        # Implement behavior prediction
        predictions = []
        
        # Example prediction logic
        for i in range(horizon):
            prediction = {
                'timestamp': current_pattern.timestamp + timedelta(minutes=i+1),
                'expected_behavior': self._predict_next_behavior(current_pattern),
                'confidence': 0.8 - (i * 0.1)  # Decreasing confidence
            }
            predictions.append(prediction)
        
        return predictions
    
    def _preprocess_features(self, features: np.ndarray) -> torch.Tensor:
        """Preprocess features for model input"""
        # Implement feature preprocessing
        features_tensor = torch.from_numpy(features).float()
        features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
        return features_tensor.to(self.device)
    
    def _generate_pattern_id(self) -> str:
        """Generate unique pattern ID"""
        return f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _update_pattern_database(self, pattern: BehaviorPattern):
        """Update pattern database with new pattern"""
        self.pattern_database.append(pattern)
        
        # Keep only recent patterns
        if len(self.pattern_database) > 1000:
            self.pattern_database = self.pattern_database[-1000:]
    
    def _predict_next_behavior(self, pattern: BehaviorPattern) -> Dict:
        """Predict next behavior based on current pattern"""
        # Implement behavior prediction logic
        return {
            'type': 'expected_behavior',
            'confidence': 0.8,
            'details': 'Predicted behavior details'
        }
    
    def export_patterns(self, filepath: str):
        """Export behavior patterns to file"""
        patterns_data = [
            {
                'pattern_id': p.pattern_id,
                'timestamp': p.timestamp.isoformat(),
                'confidence': p.confidence,
                'anomaly_score': p.anomaly_score,
                'context': p.context
            }
            for p in self.pattern_database
        ]
        
        with open(filepath, 'w') as f:
            json.dump(patterns_data, f, indent=2)
    
    def import_patterns(self, filepath: str):
        """Import behavior patterns from file"""
        with open(filepath, 'r') as f:
            patterns_data = json.load(f)
        
        self.pattern_database = [
            BehaviorPattern(
                pattern_id=p['pattern_id'],
                timestamp=datetime.fromisoformat(p['timestamp']),
                features=np.array([]),  # Features not stored in export
                confidence=p['confidence'],
                anomaly_score=p['anomaly_score'],
                context=p['context']
            )
            for p in patterns_data
        ] 