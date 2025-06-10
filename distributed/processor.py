import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import numpy as np
import torch
import tensorflow as tf
from redis.asyncio import Redis
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PROCESSED_MESSAGES = Counter('processed_messages_total', 'Total number of processed messages')
PROCESSING_TIME = Histogram('message_processing_seconds', 'Time spent processing messages')
THREAT_DETECTIONS = Counter('threat_detections_total', 'Total number of threat detections')

@dataclass
class ProcessingNode:
    id: str
    host: str
    port: int
    status: str = "active"
    load: float = 0.0

class ProcessingCluster:
    def __init__(self):
        self.redis = None
        self.kafka_consumer = None
        self.kafka_producer = None
        self.processing_tasks = {}
        self.is_running = False

    async def initialize(self):
        """Initialize the processing cluster."""
        try:
            # Initialize Redis connection
            self.redis = Redis(host='localhost', port=6379, db=0)
            await self.redis.ping()  # Test connection

            # Initialize Kafka consumer
            self.kafka_consumer = AIOKafkaConsumer(
                'threat_detection',
                bootstrap_servers='localhost:9092',
                group_id='processing_cluster'
            )

            # Initialize Kafka producer
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers='localhost:9092'
            )

            logger.info("Processing cluster initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processing cluster: {e}")
            raise

    async def start(self):
        """Start the processing cluster."""
        try:
            self.is_running = True
            await self.kafka_consumer.start()
            await self.kafka_producer.start()

            # Start processing loop
            asyncio.create_task(self._process_messages())

            logger.info("Processing cluster started successfully")
        except Exception as e:
            logger.error(f"Failed to start processing cluster: {e}")
            raise

    async def stop(self):
        """Stop the processing cluster."""
        try:
            self.is_running = False
            await self.kafka_consumer.stop()
            await self.kafka_producer.stop()
            await self.redis.close()

            # Cancel all processing tasks
            for task in self.processing_tasks.values():
                task.cancel()

            logger.info("Processing cluster stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop processing cluster: {e}")
            raise

    async def _process_messages(self):
        """Process messages from Kafka."""
        try:
            async for message in self.kafka_consumer:
                if not self.is_running:
                    break

                try:
                    # Parse message
                    data = json.loads(message.value.decode())
                    task_id = data.get('task_id')
                    
                    if not task_id:
                        logger.warning("Received message without task_id")
                        continue

                    # Create processing task
                    task = asyncio.create_task(
                        self._process_task(task_id, data)
                    )
                    self.processing_tasks[task_id] = task

                except json.JSONDecodeError:
                    logger.error("Failed to decode message")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"Error in message processing loop: {e}")
            raise

    async def _process_task(self, task_id: str, data: Dict[str, Any]):
        """Process a single task."""
        try:
            # Store task status in Redis
            await self.redis.hset(
                f"task:{task_id}",
                mapping={
                    "status": "processing",
                    "start_time": datetime.now().isoformat(),
                    "data": json.dumps(data)
                }
            )

            # Process the task
            result = await self._execute_processing(data)

            # Update task status
            await self.redis.hset(
                f"task:{task_id}",
                mapping={
                    "status": "completed",
                    "end_time": datetime.now().isoformat(),
                    "result": json.dumps(result)
                }
            )

            # Send result to Kafka
            await self.kafka_producer.send_and_wait(
                'processing_results',
                json.dumps({
                    'task_id': task_id,
                    'result': result
                }).encode()
            )

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            # Update task status to failed
            await self.redis.hset(
                f"task:{task_id}",
                mapping={
                    "status": "failed",
                    "error": str(e),
                    "end_time": datetime.now().isoformat()
                }
            )
        finally:
            # Remove task from active tasks
            self.processing_tasks.pop(task_id, None)

    async def _execute_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual processing logic."""
        try:
            # Extract data
            input_data = data.get('input_data', {})
            model_type = data.get('model_type', 'default')

            # Process based on model type
            if model_type == 'tensorflow':
                result = await self._process_tensorflow(input_data)
            elif model_type == 'pytorch':
                result = await self._process_pytorch(input_data)
            else:
                result = await self._process_default(input_data)

            return {
                'status': 'success',
                'result': result,
                'processing_time': time.time()
            }

        except Exception as e:
            logger.error(f"Error in processing execution: {e}")
            raise

    async def _process_tensorflow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using TensorFlow models."""
        try:
            # Convert input to tensor
            input_tensor = tf.convert_to_tensor(data.get('input', []))
            
            # Process using TensorFlow
            result = tf.nn.softmax(input_tensor)
            
            return {
                'type': 'tensorflow',
                'output': result.numpy().tolist()
            }
        except Exception as e:
            logger.error(f"TensorFlow processing error: {e}")
            raise

    async def _process_pytorch(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using PyTorch models."""
        try:
            # Convert input to tensor
            input_tensor = torch.tensor(data.get('input', []))
            
            # Process using PyTorch
            result = torch.nn.functional.softmax(input_tensor, dim=0)
            
            return {
                'type': 'pytorch',
                'output': result.numpy().tolist()
            }
        except Exception as e:
            logger.error(f"PyTorch processing error: {e}")
            raise

    async def _process_default(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using default methods."""
        try:
            # Convert input to numpy array
            input_array = np.array(data.get('input', []))
            
            # Process using numpy
            result = np.exp(input_array) / np.sum(np.exp(input_array))
            
            return {
                'type': 'numpy',
                'output': result.tolist()
            }
        except Exception as e:
            logger.error(f"Default processing error: {e}")
            raise

class DistributedThreatProcessor:
    def __init__(self, kafka_bootstrap_servers: List[str], redis_url: str):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.redis_url = redis_url
        self.processing_nodes: Dict[str, ProcessingNode] = {}
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.kafka_consumer: Optional[AIOKafkaConsumer] = None
        self.redis: Optional[Redis] = None
        
    async def initialize(self):
        """Initialize Kafka and Redis connections"""
        try:
            # Initialize Kafka producer
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            await self.kafka_producer.start()
            
            # Initialize Kafka consumer
            self.kafka_consumer = AIOKafkaConsumer(
                'threat-data',
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='threat-processor-group'
            )
            await self.kafka_consumer.start()
            
            # Initialize Redis connection
            self.redis = Redis(host=self.redis_url.split('//')[1].split(':')[0], port=int(self.redis_url.split(':')[2]), db=0)
            
            logger.info("Successfully initialized distributed processor")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed processor: {e}")
            raise
    
    async def process_threat_data(self, sensor_data: Dict):
        """Process incoming threat data with distributed processing"""
        start_time = datetime.now()
        
        try:
            # Distribute processing across nodes
            node = self._select_processing_node()
            
            # Send data to selected node
            await self._send_to_processing_node(node, sensor_data)
            
            # Process the data
            result = await self._process_data(sensor_data)
            
            # Update metrics
            PROCESSED_MESSAGES.inc()
            PROCESSING_TIME.observe((datetime.now() - start_time).total_seconds())
            
            if result.get('threat_detected'):
                THREAT_DETECTIONS.inc()
                await self._handle_threat_detection(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing threat data: {e}")
            raise
    
    def _select_processing_node(self) -> ProcessingNode:
        """Select the least loaded processing node"""
        if not self.processing_nodes:
            raise RuntimeError("No processing nodes available")
        
        return min(self.processing_nodes.values(), key=lambda x: x.load)
    
    async def _send_to_processing_node(self, node: ProcessingNode, data: Dict):
        """Send data to a processing node via Kafka"""
        try:
            await self.kafka_producer.send_and_wait(
                topic=f'node-{node.id}',
                value=data
            )
        except Exception as e:
            logger.error(f"Failed to send data to node {node.id}: {e}")
            raise
    
    async def _process_data(self, data: Dict) -> Dict:
        """Process the data using the distributed system"""
        # Implement actual processing logic here
        # This is a placeholder that simulates processing
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            'threat_detected': True,
            'confidence': 0.95,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_threat_detection(self, result: Dict):
        """Handle detected threats"""
        # Store threat data in Redis for quick access
        await self.redis.hset(
            f"threat:{result['timestamp']}",
            mapping={
                "status": "detected",
                "confidence": result['confidence'],
                "timestamp": result['timestamp']
            }
        )
        
        # Publish threat alert
        await self.kafka_producer.send_and_wait(
            topic='threat-alerts',
            value=result
        )
    
    async def monitor_nodes(self):
        """Monitor processing nodes health and load"""
        while True:
            try:
                for node in self.processing_nodes.values():
                    # Check node health
                    is_healthy = await self._check_node_health(node)
                    if not is_healthy:
                        node.status = "unhealthy"
                        logger.warning(f"Node {node.id} is unhealthy")
                    
                    # Update node load
                    node.load = await self._get_node_load(node)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring nodes: {e}")
                await asyncio.sleep(1)
    
    async def _check_node_health(self, node: ProcessingNode) -> bool:
        """Check if a processing node is healthy"""
        try:
            # Implement actual health check logic
            return True
        except Exception:
            return False
    
    async def _get_node_load(self, node: ProcessingNode) -> float:
        """Get the current load of a processing node"""
        try:
            # Implement actual load calculation
            return 0.0
        except Exception:
            return 1.0  # Return high load on error
    
    async def shutdown(self):
        """Gracefully shutdown the processor"""
        if self.kafka_producer:
            await self.kafka_producer.stop()
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        if self.redis:
            await self.redis.close()

async def main():
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Initialize processor
    processor = DistributedThreatProcessor(
        kafka_bootstrap_servers=['localhost:9092'],
        redis_url='redis://localhost'
    )
    
    try:
        await processor.initialize()
        
        # Start node monitoring
        asyncio.create_task(processor.monitor_nodes())
        
        # Process messages
        async for message in processor.kafka_consumer:
            await processor.process_threat_data(message.value)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await processor.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 