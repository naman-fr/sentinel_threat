import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
import logging
import threading
import time
import queue
from aiokafka import AIOKafkaConsumer
import redis.asyncio as redis
import plotly.express as px
import plotly.graph_objects as go
from collections import deque
from typing import Dict, Deque, Any, Optional
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreatDashboard:
    def __init__(self, kafka_bootstrap_servers: list, redis_url: str):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.redis_url = redis_url
        self.consumer = None
        self.redis = None
        self._consumer_thread = None
        self._stop_event = threading.Event()
        self._consumer_loop = None
        self.message_queue = queue.Queue()

    async def initialize_connections(self):
        """Initialize dashboard connections (excluding Redis, now in consumer thread)."""
        logger.info("Initializing dashboard connections...")
        # Redis initialization moved to _kafka_consumer_thread_target
        logger.info("Successfully initialized dashboard connections")

    def _kafka_consumer_thread_target(self):
        """Target function for the Kafka consumer thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._consumer_loop = loop # Store the loop instance

        async def run_consumer_loop_async():
            # Initialize Redis connection within this thread's event loop
            try:
                self.redis = await redis.from_url(self.redis_url)
                await self.redis.ping()
                logger.info("Redis connection successful within consumer thread.")
            except Exception as e:
                logger.error(f"Failed to connect to Redis in consumer thread: {e}")
                # This exception needs to be handled gracefully to not stop the consumer thread

            self.consumer = AIOKafkaConsumer(
                'threat_detection',
                bootstrap_servers=self.kafka_bootstrap_servers,
                group_id='dashboard_group',
                auto_offset_reset='latest'
            )

            try:
                await self.consumer.start()
                logger.info("Kafka consumer started in background thread.")

                while not self._stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(
                            self.consumer.getone(),
                            timeout=1.0
                        )
                        if message:
                            try:
                                data = json.loads(message.value.decode())
                                logger.info(f"Received message: {data}")
                                self.message_queue.put(data)
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to decode message: {e}")
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error getting message in consumer loop: {e}")
                        await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Kafka consumer thread startup or main loop error: {e}")
            finally:
                logger.info("Consumer loop stopping.")
                if self.consumer and not self.consumer._closed:
                    try:
                        await self.consumer.stop()
                        logger.info("Kafka consumer stopped.")
                    except Exception as e:
                        logger.error(f"Error stopping consumer in thread: {e}")
                
                if self.redis:
                    try:
                        await self.redis.aclose()
                        logger.info("Redis connection closed in consumer thread.")
                    except Exception as e:
                        logger.error(f"Error closing Redis in consumer thread: {e}")
                
                # The loop will be closed outside this async function.

        try:
            loop.run_until_complete(run_consumer_loop_async())
        except Exception as e:
            logger.error(f"Error in consumer thread: {e}")
        finally:
            if loop and not loop.is_closed():
                loop.close()
                logger.info("Consumer thread event loop closed.")

    def start_kafka_consumer(self):
        """Start Kafka consumer in background thread."""
        self._stop_event.clear()
        self._consumer_thread = threading.Thread(
            target=self._kafka_consumer_thread_target,
            daemon=True
        )
        self._consumer_thread.start()
        logger.info("Kafka consumer thread initiated.")

    def stop(self):
        """Stop the dashboard and cleanup resources"""
        logger.info("Stopping dashboard...")
        self._stop_event.set()

        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=10.0)
            if self._consumer_thread.is_alive():
                logger.warning("Consumer thread did not terminate gracefully within timeout.")

        # Redis cleanup is now handled in _kafka_consumer_thread_target's finally block
        # if self.redis:
        #     try:
        #         # No need to ping, just attempt to close if it exists
        #         await self.redis.aclose()
        #         logger.info("Redis connection closed.")
        #     except Exception as e:
        #         logger.error(f"Error closing Redis connection: {e}")
        
        logger.info("Dashboard stopped.")

    def render_dashboard(self):
        """Render the main dashboard interface"""
        try:
            # Header
            st.title("🛡️ Sentinel Threat Detection Dashboard")
            st.markdown("Real-time threat monitoring and analysis")
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                self._render_threat_count(col1)
            with col2:
                self._render_severity_metrics(col2)
            with col3:
                self._render_response_time(col3)
            with col4:
                self._render_system_health(col4)
            
            # Main content
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                self._render_threat_map()
                self._render_threat_timeline()
            
            with col_right:
                self._render_alerts_panel()
                self._render_threat_details()
            
        except Exception as e:
            logger.error(f"Error rendering dashboard: {e}")
            st.error("Error rendering dashboard. Please check logs for details.")
    
    def _render_threat_count(self, parent_col):
        """Render threat count metrics"""
        try:
            with parent_col:
                st.metric(
                    label="Active Threats",
                    value=len(st.session_state.get('alerts', [])),
                    delta=None
                )
        except Exception as e:
            logger.error(f"Error rendering threat count: {e}")
            st.error("Error displaying metrics")
    
    def _render_severity_metrics(self, parent_col):
        """Render threat severity metrics"""
        avg_severity = np.mean([a['severity'] for a in st.session_state.get('alerts', [])]) if st.session_state.get('alerts') else 0
        with parent_col:
            fig_severity = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_severity,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Threat Severity"},
                gauge={
                    'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.5], 'color': 'lightgray'},
                        {'range': [0.5, 1], 'color': 'gray'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ))
            fig_severity.update_layout(margin=dict(l=20, r=20, t=30, b=30))
            st.plotly_chart(fig_severity, use_container_width=True, key='threat_severity_gauge_chart')
    
    def _render_response_time(self, parent_col):
        """Render response time metrics"""
        response_times = self._get_response_times()
        with parent_col:
            st.metric(
                label="Avg Response Time",
                value=f"{response_times['average']:.2f}s",
                delta=f"{response_times['delta']:.2f}s"
            )
    
    def _render_system_health(self, parent_col):
        """Render system health metrics"""
        system_health_value = 100 if (self._consumer_thread and self._consumer_thread.is_alive()) else 0
        with parent_col:
            fig_health = go.Figure(go.Indicator(
                mode="gauge+number",
                value=system_health_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgreen"},
                    'bar': {'color': "darkgreen"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'red'},
                        {'range': [50, 75], 'color': 'orange'},
                        {'range': [75, 100], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ))
            fig_health.update_layout(margin=dict(l=20, r=20, t=30, b=30))
            st.plotly_chart(fig_health, use_container_width=True, key='system_health_gauge_chart')
    
    def _render_threat_map(self):
        """Render threat map"""
        try:
            df = self._get_threat_locations()
            if not df.empty:
                fig = px.scatter_geo(df, 
                                   lat='latitude', 
                                   lon='longitude',
                                   color='severity', 
                                   size='confidence',
                                   hover_name='type', 
                                   title='Threat Locations',
                                   color_continuous_scale='Viridis',
                                   projection='natural earth')
                fig.update_layout(
                    geo=dict(
                        showland=True,
                        landcolor='rgb(243, 243, 243)',
                        countrycolor='rgb(204, 204, 204)',
                    ),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No threat data available for map visualization.")
        except Exception as e:
            logger.error(f"Error rendering threat map: {e}")
            st.error("Error rendering threat map. Please check logs for details.")
    
    def _render_threat_timeline(self):
        """Render threat timeline"""
        try:
            df = self._get_timeline_data()
            if not df.empty:
                # Convert timestamp strings to datetime objects
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                fig = px.line(df, 
                             x='timestamp', 
                             y='severity',
                             color='type', 
                             title='Threat Severity Over Time',
                             labels={'timestamp': 'Time', 'severity': 'Severity', 'type': 'Threat Type'})
                
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Severity",
                    legend_title="Threat Type",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No threat data available for timeline visualization.")
        except Exception as e:
            logger.error(f"Error rendering threat timeline: {e}")
            st.error("Error rendering threat timeline. Please check logs for details.")
    
    def _render_alerts_panel(self):
        """Render real-time alerts panel"""
        st.subheader("Active Alerts")
        
        if not st.session_state.alerts:
            st.info("No active alerts")
            return
            
        for alert in st.session_state.alerts:
            with st.expander(f"Alert: {alert['type']} - {alert['id']}"):
                st.write(f"ID: {alert['id']}")
                st.write(f"Severity: {alert['severity']:.2f}")
                st.write(f"Confidence: {alert['confidence']:.2f}")
                st.write(f"Location: {alert['location']}")
                st.write(f"Time: {alert['timestamp']}")
                
                if st.button("Acknowledge", key=f"ack_{alert['id']}"):
                    self._acknowledge_alert(alert['id'])
    
    def _render_threat_details(self):
        """Render detailed threat information"""
        st.subheader("Threat Details")
        
        if not st.session_state.threat_history:
            st.info("No threat history available")
            return
            
        selected_threat = st.selectbox(
            "Select Threat",
            options=[t['id'] for t in st.session_state.threat_history]
        )
        
        if selected_threat:
            threat_data = self._get_threat_details(selected_threat)
            
            st.write("### Threat Analysis")
            st.write(f"Type: {threat_data['type']}")
            st.write(f"Severity: {threat_data['severity']}")
            st.write(f"Confidence: {threat_data['confidence']}")
            
            st.write("### Behavioral Analysis")
            st.write(threat_data['behavioral_analysis'])
            
            st.write("### Recommended Actions")
            for action in threat_data['recommended_actions']:
                st.write(f"- {action}")
    
    def _render_confidence_metric(self, parent_col):
        """Render average confidence metric"""
        try:
            avg_confidence = np.mean([a['confidence'] for a in st.session_state.get('alerts', [])]) if st.session_state.get('alerts') else 0
            with parent_col:
                st.metric(
                    label="Avg Confidence",
                    value=f"{avg_confidence:.2f}",
                    delta=None
                )
        except Exception as e:
            logger.error(f"Error rendering average confidence: {e}")
            st.error("Error displaying confidence metric")
    
    def _calculate_threat_delta(self) -> int:
        """Calculate change in threat count"""
        if len(st.session_state.threat_history) < 2:
            return 0
        return len(st.session_state.alerts) - len(st.session_state.threat_history[-2]['alerts'])
    
    def _get_severity_data(self) -> Dict:
        """Get threat severity metrics"""
        if not st.session_state.alerts:
            return {'average': 0, 'max': 0, 'min': 0}
        
        severities = [alert['severity'] for alert in st.session_state.alerts]
        return {
            'average': np.mean(severities),
            'max': max(severities),
            'min': min(severities)
        }
    
    def _get_response_times(self) -> Dict:
        """Calculate and return response time metrics"""
        if not st.session_state.get('response_times_history'):
            return {'average': 0.0, 'delta': 0.0}

        response_times = list(st.session_state.response_times_history)
        
        current_average = np.mean(response_times) if response_times else 0.0
        
        # Simple delta calculation: change from previous average if enough data
        delta = 0.0
        if len(response_times) > 1:
            previous_average = np.mean(response_times[:-1]) # Average of all but the last one
            delta = current_average - previous_average
            
        return {'average': current_average, 'delta': delta}
    
    def _get_system_health(self) -> Dict:
        """Get system health metrics"""
        health_score = 95  # Example value
        return {
            'health_score': health_score,
            'color': 'green' if health_score > 90 else 'orange'
        }
    
    def _get_threat_locations(self) -> pd.DataFrame:
        """Get threat location data"""
        logger.info(f"Current alerts in session state: {len(st.session_state.alerts)}")
        
        if not st.session_state.alerts:
            return pd.DataFrame({
                'latitude': [],
                'longitude': [],
                'severity': [],
                'confidence': [],
                'type': [],
                'timestamp': []
            })
        
        try:
            df = pd.DataFrame([{
                'latitude': alert['location']['latitude'],
                'longitude': alert['location']['longitude'],
                'severity': alert['severity'],
                'confidence': alert['confidence'],
                'type': alert['type'],
                'timestamp': alert['timestamp']
            } for alert in st.session_state.alerts])
            
            logger.info(f"Created DataFrame with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error creating threat locations DataFrame: {e}")
            return pd.DataFrame({
                'latitude': [],
                'longitude': [],
                'severity': [],
                'confidence': [],
                'type': [],
                'timestamp': []
            })
    
    def _get_timeline_data(self) -> pd.DataFrame:
        """Get threat timeline data"""
        if not st.session_state.alerts:
            return pd.DataFrame({
                'timestamp': [],
                'severity': [],
                'type': []
            })
            
        try:
            # Create DataFrame from alerts
            df = pd.DataFrame([{
                'timestamp': alert['timestamp'],
                'severity': alert['severity'],
                'type': alert['type']
            } for alert in st.session_state.alerts])
            
            # Convert timestamp strings to datetime objects
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            return df
        except Exception as e:
            logger.error(f"Error creating timeline data: {e}")
            return pd.DataFrame({
                'timestamp': [],
                'severity': [],
                'type': []
            })
    
    def _get_threat_details(self, threat_id: str) -> Dict:
        """Get detailed threat information"""
        # Find the threat in history
        for history in st.session_state.threat_history:
            if history.get('id') == threat_id:
                return {
                    'type': history.get('type', 'Unknown'),
                    'severity': history.get('severity', 0),
                    'confidence': history.get('confidence', 0),
                    'behavioral_analysis': f"Analysis of {history.get('type', 'Unknown')} threat with severity {history.get('severity', 0)}",
                    'recommended_actions': [
                        "Monitor affected systems",
                        "Review security logs",
                        "Update threat signatures"
                    ]
                }
        return {
            'type': 'Unknown',
            'severity': 0,
            'confidence': 0,
            'behavioral_analysis': 'No analysis available',
            'recommended_actions': ['No actions available']
        }
    
    def _acknowledge_alert(self, alert_id: str):
        """Acknowledge and handle alert"""
        st.session_state.alerts = [a for a in st.session_state.alerts if a['id'] != alert_id]
        st.rerun()

    def _process_message(self, message):
        """Process a single message from the queue"""
        try:
            # Add task ID if not present
            if 'task_id' not in message:
                message['task_id'] = f"task_{int(time.time() * 1000)}"
            
            # Add message to alerts
            st.session_state.alerts.append(message)
            
            # Keep only last 100 alerts
            if len(st.session_state.alerts) > 100:
                st.session_state.alerts = deque(list(st.session_state.alerts)[-100:])
            
            # Add to threat history
            st.session_state.threat_history.append({
                'id': message.get('id', f"threat_{int(time.time() * 1000)}"),
                'task_id': message['task_id'],
                'timestamp': message.get('timestamp', pd.Timestamp.now().isoformat()),
                'type': message.get('type', 'Unknown'),
                'severity': message.get('severity', 0.0),
                'confidence': message.get('confidence', 0.0),
                'location': message.get('location', {})
            })
            
            # Keep only last 1000 history entries
            if len(st.session_state.threat_history) > 1000:
                st.session_state.threat_history = deque(list(st.session_state.threat_history)[-1000:])
            
            # Calculate and store response times
            if len(st.session_state.threat_history) > 1:
                current_timestamp = pd.to_datetime(message.get('timestamp', pd.Timestamp.now().isoformat()))
                previous_threat = st.session_state.threat_history[-2]
                previous_timestamp = pd.to_datetime(previous_threat.get('timestamp', pd.Timestamp.now().isoformat()))
                response_time = (current_timestamp - previous_timestamp).total_seconds()
                st.session_state.response_times_history.append(response_time)
            
            logger.info(f"Processed message with task_id: {message['task_id']}")
            return True
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return False

def main():
    """Main entry point for the Streamlit dashboard"""
    try:
        # Set page config
        st.set_page_config(
            page_title="Threat Detection Dashboard",
            page_icon="🛡️",
            layout="wide"
        )
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.alerts = deque(maxlen=100)
            st.session_state.threat_history = deque(maxlen=1000)
            st.session_state.response_times_history = deque(maxlen=100)
            st.session_state.dashboard = None
        
        # Initialize dashboard if not already done
        if not st.session_state.initialized or st.session_state.dashboard is None:
            try:
                dashboard = ThreatDashboard(
                    kafka_bootstrap_servers=["localhost:9092"],
                    redis_url="redis://localhost"
                )
                st.session_state.dashboard = dashboard
                st.session_state.initialized = True
                dashboard.start_kafka_consumer()
                logger.info("Dashboard initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize dashboard: {e}")
                st.error("Failed to initialize dashboard. Please check logs for details.")
                return
        
        # Process messages from the queue
        current_dashboard = st.session_state.dashboard
        messages_processed_this_run = False
        
        # Process all available messages
        while not current_dashboard.message_queue.empty():
            try:
                message = current_dashboard.message_queue.get_nowait()
                if current_dashboard._process_message(message):
                    messages_processed_this_run = True
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing message from queue: {e}")
        
        # Render dashboard
        try:
            current_dashboard.render_dashboard()
        except Exception as e:
            logger.error(f"Dashboard rendering error: {e}")
            st.error("Error rendering dashboard. Please check logs for details.")
        
        # If new messages were processed, force a rerun to update the UI
        if messages_processed_this_run:
            st.rerun()
            
    except Exception as e:
        logger.error(f"Main error: {e}")
        st.error("An error occurred. Please check logs for details.")

if __name__ == "__main__":
    main() 