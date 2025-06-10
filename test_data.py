import asyncio
import json
import random
from datetime import datetime
from aiokafka import AIOKafkaProducer

async def send_test_data():
    producer = AIOKafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    await producer.start()
    
    try:
        while True:
            # Generate random threat data
            threat_data = {
                'task_id': f"task_{random.randint(1000, 9999)}",
                'id': f"threat_{random.randint(1000, 9999)}",
                'type': random.choice(['Malware', 'Phishing', 'DDoS', 'Data Exfiltration']),
                'severity': random.uniform(0.1, 1.0),
                'confidence': random.uniform(0.5, 1.0),
                'location': {
                    'latitude': random.uniform(-90, 90),
                    'longitude': random.uniform(-180, 180)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to Kafka
            await producer.send_and_wait('threat_detection', threat_data)
            print(f"Sent threat data: {threat_data}")
            
            # Wait before sending next data
            await asyncio.sleep(2)
            
    except KeyboardInterrupt:
        print("Stopping test data generation...")
    finally:
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(send_test_data()) 