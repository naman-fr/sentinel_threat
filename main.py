import asyncio
import logging
from pathlib import Path
import yaml
import tracemalloc
from security.zero_trust import ZeroTrustManager
from security.integrations import SecurityIntegrator
from ai.behavioral_analysis import BehavioralAnalyzer
from distributed.processor import ProcessingCluster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable tracemalloc for memory tracking
tracemalloc.start()

class ThreatDetectionSystem:
    def __init__(self, config_path: str = "security/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.zero_trust = None
        self.security_integrator = None
        self.behavior_analyzer = None
        self.processing_cluster = None

    def _load_config(self) -> dict:
        """Load configuration from YAML file and decode bytes to strings if needed."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Recursively decode bytes to strings
            def decode_bytes(obj):
                if isinstance(obj, dict):
                    return {decode_bytes(k): decode_bytes(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [decode_bytes(i) for i in obj]
                elif isinstance(obj, bytes):
                    return obj.decode('utf-8')
                else:
                    return obj
            return decode_bytes(config)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    async def initialize(self):
        """Initialize the system"""
        try:
            # Load configuration
            self.config = self._load_config()
            
            # Initialize components
            self.zero_trust = ZeroTrustManager()
            await self.zero_trust.initialize()
            
            # Initialize Security Integrator
            self.security_integrator = SecurityIntegrator()
            await self.security_integrator.initialize()

            # Initialize Behavior Analyzer
            self.behavior_analyzer = BehavioralAnalyzer()
            await self.behavior_analyzer.initialize()

            # Initialize Processing Cluster
            self.processing_cluster = ProcessingCluster()
            await self.processing_cluster.initialize()

            logger.info("System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise

    async def start(self):
        """Start the threat detection system."""
        try:
            # Start processing cluster
            await self.processing_cluster.start()

            logger.info("System started successfully")
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise

    async def stop(self):
        """Stop the threat detection system."""
        try:
            # Stop all components
            await self.processing_cluster.stop()
            await self.security_integrator.stop_monitoring()
            await self.behavior_analyzer.stop_analysis()

            logger.info("System stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
            raise

async def main():
    """Main entry point for the application."""
    try:
        # Create data directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)

        # Initialize and start the system
        system = ThreatDetectionSystem()
        await system.initialize()
        await system.start()

        # Keep the system running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down system...")
            await system.stop()

    except Exception as e:
        logger.error(f"System error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 