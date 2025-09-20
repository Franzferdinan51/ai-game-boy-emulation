#!/usr/bin/env python3
"""
Enhanced Service Monitoring System for PyGB Project
Monitors backend (port 5000) and frontend (port 5173) services with real-time dashboard,
auto-restart capabilities, resource monitoring, and comprehensive logging
"""

import asyncio
import json
import logging
import os
import psutil
import requests
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import signal
import sys
from enum import Enum
from collections import deque
import socket
import urllib3
import warnings

# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

class ServiceStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    CRASHED = "crashed"
    RESTARTING = "restarting"
    UNKNOWN = "unknown"

@dataclass
class ServiceInfo:
    name: str
    port: int
    process: Optional[subprocess.Popen] = None
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_check: datetime = None
    last_restart: datetime = None
    restart_count: int = 0
    health_check_url: str = ""
    start_command: str = ""
    working_dir: str = ""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    response_time: float = 0.0
    uptime: float = 0.0
    error_count: int = 0
    last_error: str = ""
    log_file: str = ""
    auto_restart: bool = True
    max_restarts: int = 5
    restart_cooldown: int = 60

@dataclass
class AlertEvent:
    timestamp: datetime
    service_name: str
    event_type: str
    message: str
    severity: str  # "info", "warning", "error", "critical"
    details: Dict = None

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_total: float
    cpu_per_core: List[float]
    memory_total: float
    memory_used: float
    memory_percent: float
    disk_total: float
    disk_used: float
    disk_percent: float
    network_sent: float
    network_received: float
    uptime: float
    process_count: int

class ServiceMonitor:
    def __init__(self, config_path: str = "monitor_config.json"):
        self.services: Dict[str, ServiceInfo] = {}
        self.alerts: List[AlertEvent] = []
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.running = False
        self.monitor_thread = None
        self.web_server_thread = None
        self.terminal_dashboard_thread = None
        self.max_alerts = 1000
        self.health_check_interval = 10  # seconds
        self.resource_check_interval = 5  # seconds
        self.max_restart_attempts = 5
        self.restart_cooldown = 60  # seconds
        self.cpu_threshold = 80  # percent
        self.memory_threshold = 80  # percent
        self.enable_terminal_dashboard = True
        self.enable_web_dashboard = True

        # Network stats tracking
        self.last_network_stats = None

        # Initialize services
        self._initialize_services()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("service_monitor")
        logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(
            log_dir / f"service_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        default_config = {
            "services": {
                "backend": {
                    "port": 5000,
                    "health_check_url": "http://localhost:5000/health",
                    "status_check_url": "http://localhost:5000/api/status",
                    "start_command": "python src/main.py",
                    "working_dir": "ai-game-server",
                    "auto_restart": True,
                    "max_restarts": 5,
                    "restart_cooldown": 60
                },
                "frontend": {
                    "port": 5173,
                    "health_check_url": "http://localhost:5173",
                    "start_command": "npm run dev",
                    "working_dir": "ai-game-assistant",
                    "auto_restart": True,
                    "max_restarts": 5,
                    "restart_cooldown": 60
                }
            },
            "monitoring": {
                "health_check_interval": 10,
                "resource_check_interval": 5,
                "max_restart_attempts": 5,
                "restart_cooldown": 60,
                "cpu_threshold": 80,
                "memory_threshold": 80,
                "max_alerts": 1000,
                "system_metrics_history": 1000
            },
            "dashboards": {
                "terminal": {
                    "enabled": True,
                    "refresh_rate": 2
                },
                "web": {
                    "enabled": True,
                    "port": 8080,
                    "auto_refresh": 10
                }
            },
            "logging": {
                "level": "INFO",
                "file": "logs/service_monitor.log",
                "max_size": "10MB",
                "backup_count": 5
            },
            "alerts": {
                "enable_email": False,
                "enable_slack": False,
                "email_recipients": [],
                "slack_webhook": ""
            }
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Deep merge with default config
                    def deep_merge(base, update):
                        for key, value in update.items():
                            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                                deep_merge(base[key], value)
                            else:
                                base[key] = value
                    deep_merge(default_config, user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def _initialize_services(self):
        """Initialize service configurations"""
        monitoring_config = self.config.get("monitoring", {})
        dashboards_config = self.config.get("dashboards", {})

        self.health_check_interval = monitoring_config.get("health_check_interval", 10)
        self.resource_check_interval = monitoring_config.get("resource_check_interval", 5)
        self.max_restart_attempts = monitoring_config.get("max_restart_attempts", 5)
        self.restart_cooldown = monitoring_config.get("restart_cooldown", 60)
        self.cpu_threshold = monitoring_config.get("cpu_threshold", 80)
        self.memory_threshold = monitoring_config.get("memory_threshold", 80)
        self.max_alerts = monitoring_config.get("max_alerts", 1000)

        self.enable_terminal_dashboard = dashboards_config.get("terminal", {}).get("enabled", True)
        self.enable_web_dashboard = dashboards_config.get("web", {}).get("enabled", True)

        for service_name, service_config in self.config.get("services", {}).items():
            service = ServiceInfo(
                name=service_name,
                port=service_config.get("port", 0),
                health_check_url=service_config.get("health_check_url", ""),
                start_command=service_config.get("start_command", ""),
                working_dir=service_config.get("working_dir", ""),
                auto_restart=service_config.get("auto_restart", True),
                max_restarts=service_config.get("max_restarts", 5),
                restart_cooldown=service_config.get("restart_cooldown", 60)
            )
            self.services[service_name] = service
            self.logger.info(f"Initialized service: {service_name}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def add_alert(self, service_name: str, event_type: str, message: str, severity: str = "info"):
        """Add an alert event"""
        alert = AlertEvent(
            timestamp=datetime.now(),
            service_name=service_name,
            event_type=event_type,
            message=message,
            severity=severity
        )

        self.alerts.append(alert)

        # Keep only the most recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

        # Log the alert
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(f"[{service_name}] {event_type}: {message}")

    def check_service_health(self, service: ServiceInfo) -> bool:
        """Check if a service is healthy"""
        try:
            start_time = time.time()
            is_healthy = False

            # Try health check URL first
            if service.health_check_url:
                try:
                    response = requests.get(
                        service.health_check_url,
                        timeout=5,
                        verify=False
                    )
                    service.response_time = (time.time() - start_time) * 1000  # ms
                    is_healthy = response.status_code == 200
                except requests.RequestException:
                    pass

            # Fallback to port check
            if not is_healthy:
                is_healthy = self._check_port(service.port)
                service.response_time = (time.time() - start_time) * 1000 if is_healthy else 0

            return is_healthy
        except Exception as e:
            self.logger.debug(f"Health check error for {service.name}: {e}")
            return False

    def _check_port(self, port: int) -> bool:
        """Check if a port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def get_system_metrics(self) -> SystemMetrics:
        """Get comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_total = psutil.cpu_percent(interval=1)
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_total = memory.total / (1024**3)  # GB
            memory_used = memory.used / (1024**3)  # GB
            memory_percent = memory.percent

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_total = disk.total / (1024**3)  # GB
            disk_used = disk.used / (1024**3)  # GB
            disk_percent = disk.percent

            # Network metrics
            network = psutil.net_io_counters()
            network_sent = network.bytes_sent / (1024**2)  # MB
            network_received = network.bytes_recv / (1024**2)  # MB

            # System uptime
            uptime = time.time() - psutil.boot_time()

            # Process count
            process_count = len(psutil.pids())

            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_total=cpu_total,
                cpu_per_core=cpu_per_core,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_percent=memory_percent,
                disk_total=disk_total,
                disk_used=disk_used,
                disk_percent=disk_percent,
                network_sent=network_sent,
                network_received=network_received,
                uptime=uptime,
                process_count=process_count
            )
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return None

    def get_process_resources(self, service: ServiceInfo) -> Tuple[float, float, float]:
        """Get resource usage for a service process"""
        if not service.process:
            return 0.0, 0.0, 0.0

        try:
            # Get all child processes
            process = psutil.Process(service.process.pid)
            processes = [process] + process.children(recursive=True)

            total_cpu = 0.0
            total_memory = 0.0

            for proc in processes:
                try:
                    total_cpu += proc.cpu_percent(interval=0.1)
                    total_memory += proc.memory_info().rss / 1024 / 1024  # MB
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Calculate memory percentage
            total_memory_percent = (total_memory / psutil.virtual_memory().total) * 100 * 1024

            return total_cpu, total_memory_percent, total_memory
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0, 0.0, 0.0

    def get_process_resources(self, service: ServiceInfo) -> Tuple[float, float, float]:
        """Get resource usage for a service process"""
        if not service.process:
            return 0.0, 0.0, 0.0

        try:
            # Get all child processes
            process = psutil.Process(service.process.pid)
            processes = [process] + process.children(recursive=True)

            total_cpu = 0.0
            total_memory = 0.0

            for proc in processes:
                try:
                    total_cpu += proc.cpu_percent(interval=0.1)
                    total_memory += proc.memory_info().rss / 1024 / 1024  # MB
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Calculate memory percentage
            total_memory_percent = (total_memory / psutil.virtual_memory().total) * 100 * 1024

            return total_cpu, total_memory_percent, total_memory
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0, 0.0, 0.0

    def start_service(self, service: ServiceInfo) -> bool:
        """Start a service"""
        try:
            if service.status == ServiceStatus.RUNNING:
                self.logger.info(f"Service {service.name} is already running")
                return True

            self.add_alert(service.name, "START_ATTEMPT", f"Starting service {service.name}")

            # Change to working directory
            old_cwd = os.getcwd()
            os.chdir(service.working_dir)

            # Start the process
            service.process = subprocess.Popen(
                service.start_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )

            os.chdir(old_cwd)

            # Wait a bit for the service to start
            time.sleep(5)

            # Check if it's running
            if self.check_service_health(service):
                service.status = ServiceStatus.RUNNING
                service.last_restart = datetime.now()
                service.restart_count += 1
                self.add_alert(service.name, "START_SUCCESS", f"Service {service.name} started successfully", "info")
                return True
            else:
                service.status = ServiceStatus.CRASHED
                self.add_alert(service.name, "START_FAILED", f"Service {service.name} failed to start", "error")
                return False

        except Exception as e:
            service.status = ServiceStatus.CRASHED
            self.add_alert(service.name, "START_ERROR", f"Error starting service {service.name}: {e}", "error")
            return False

    def stop_service(self, service: ServiceInfo) -> bool:
        """Stop a service"""
        try:
            if service.process:
                service.process.terminate()
                service.process.wait(timeout=10)
                service.process = None

            service.status = ServiceStatus.STOPPED
            self.add_alert(service.name, "STOP_SUCCESS", f"Service {service.name} stopped", "info")
            return True
        except Exception as e:
            self.add_alert(service.name, "STOP_ERROR", f"Error stopping service {service.name}: {e}", "error")
            return False

    def restart_service(self, service: ServiceInfo) -> bool:
        """Restart a service"""
        self.add_alert(service.name, "RESTART_ATTEMPT", f"Restarting service {service.name}")

        # Check if we've exceeded restart attempts
        if service.restart_count >= self.max_restart_attempts:
            time_since_last_restart = (datetime.now() - service.last_restart).total_seconds()
            if time_since_last_restart < self.restart_cooldown:
                self.add_alert(
                    service.name,
                    "RESTART_LIMIT",
                    f"Restart limit reached for {service.name}. Waiting for cooldown.",
                    "warning"
                )
                return False

        # Stop the service
        self.stop_service(service)
        time.sleep(2)

        # Start the service
        return self.start_service(service)

    def monitor_services(self):
        """Main monitoring loop"""
        self.logger.info("Starting service monitoring...")

        while self.running:
            try:
                for service in self.services.values():
                    # Check service health
                    old_status = service.status
                    is_healthy = self.check_service_health(service)

                    if is_healthy:
                        service.status = ServiceStatus.RUNNING
                        if old_status != ServiceStatus.RUNNING:
                            self.add_alert(service.name, "HEALTHY", f"Service {service.name} is healthy", "info")
                    else:
                        if old_status == ServiceStatus.RUNNING:
                            service.status = ServiceStatus.CRASHED
                            self.add_alert(service.name, "UNHEALTHY", f"Service {service.name} is unhealthy", "warning")

                            # Try to restart the service
                            if self.restart_service(service):
                                self.add_alert(service.name, "RESTART_SUCCESS", f"Service {service.name} restarted successfully", "info")
                            else:
                                self.add_alert(service.name, "RESTART_FAILED", f"Failed to restart service {service.name}", "error")

                    service.last_check = datetime.now()

                    # Get resource usage
                    cpu_percent, memory_percent, memory_mb = self.get_process_resources(service)
                    service.cpu_percent = cpu_percent
                    service.memory_percent = memory_percent
                    service.memory_mb = memory_mb

                    # Check resource thresholds
                    if cpu_percent > self.cpu_threshold:
                        self.add_alert(service.name, "HIGH_CPU", f"High CPU usage: {cpu_percent:.1f}%", "warning")

                    if memory_percent > self.memory_threshold:
                        self.add_alert(service.name, "HIGH_MEMORY", f"High memory usage: {memory_percent:.1f}%", "warning")

                # Sleep for the health check interval
                time.sleep(self.health_check_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)

    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "running": self.running,
            "services": {
                name: {
                    "name": service.name,
                    "port": service.port,
                    "status": service.status.value,
                    "last_check": service.last_check.isoformat() if service.last_check else None,
                    "last_restart": service.last_restart.isoformat() if service.last_restart else None,
                    "restart_count": service.restart_count,
                    "cpu_percent": service.cpu_percent,
                    "memory_percent": service.memory_percent,
                    "memory_mb": service.memory_mb,
                    "uptime": self._calculate_uptime(service)
                }
                for name, service in self.services.items()
            },
            "alerts": [asdict(alert) for alert in self.alerts[-10:]],  # Last 10 alerts
            "system_info": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                "memory_available": psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
                "disk_usage": psutil.disk_usage('/').percent
            }
        }

    def _calculate_uptime(self, service: ServiceInfo) -> Optional[float]:
        """Calculate service uptime in seconds"""
        if service.status != ServiceStatus.RUNNING or not service.last_restart:
            return None
        return (datetime.now() - service.last_restart).total_seconds()

    def start(self):
        """Start the monitoring system"""
        if self.running:
            self.logger.warning("Monitor is already running")
            return

        self.running = True
        self.logger.info("Starting service monitoring system...")

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        self.monitor_thread.start()

        # Start web dashboard if enabled
        if self.config.get("web_dashboard", {}).get("enabled", True):
            self.web_server_thread = threading.Thread(target=self._start_web_dashboard, daemon=True)
            self.web_server_thread.start()

        # Initial service start
        for service in self.services.values():
            self.start_service(service)

        self.logger.info("Service monitoring system started")

    def stop(self):
        """Stop the monitoring system"""
        if not self.running:
            return

        self.logger.info("Stopping service monitoring system...")
        self.running = False

        # Stop all services
        for service in self.services.values():
            self.stop_service(service)

        # Wait for threads to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        if self.web_server_thread:
            self.web_server_thread.join(timeout=5)

        self.logger.info("Service monitoring system stopped")

    def _start_web_dashboard(self):
        """Start the web dashboard"""
        try:
            from flask import Flask, jsonify, render_template_string

            app = Flask(__name__)

            # HTML template for the dashboard
            HTML_TEMPLATE = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Service Monitor Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .header { background-color: #333; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                    .service-card { background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                    .status-running { color: #28a745; font-weight: bold; }
                    .status-stopped { color: #dc3545; font-weight: bold; }
                    .status-crashed { color: #fd7e14; font-weight: bold; }
                    .status-restarting { color: #ffc107; font-weight: bold; }
                    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }
                    .metric { background-color: #f8f9fa; padding: 10px; border-radius: 3px; text-align: center; }
                    .metric-value { font-size: 24px; font-weight: bold; color: #333; }
                    .metric-label { font-size: 12px; color: #666; }
                    .alerts { background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                    .alert { padding: 10px; margin-bottom: 10px; border-radius: 3px; }
                    .alert-info { background-color: #d1ecf1; border-left: 4px solid #17a2b8; }
                    .alert-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; }
                    .alert-error { background-color: #f8d7da; border-left: 4px solid #dc3545; }
                    .system-info { background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-top: 20px; }
                    .refresh-info { text-align: center; margin-top: 20px; color: #666; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Service Monitor Dashboard</h1>
                        <p>Real-time monitoring of backend and frontend services</p>
                    </div>

                    <div id="services-container"></div>

                    <div class="alerts">
                        <h3>Recent Alerts</h3>
                        <div id="alerts-container"></div>
                    </div>

                    <div class="system-info">
                        <h3>System Information</h3>
                        <div id="system-info-container"></div>
                    </div>

                    <div class="refresh-info">
                        <p>Page auto-refreshes every 10 seconds</p>
                        <p>Last updated: <span id="last-updated">Never</span></p>
                    </div>
                </div>

                <script>
                    function getStatus() {
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {
                                updateServices(data.services);
                                updateAlerts(data.alerts);
                                updateSystemInfo(data.system_info);
                                document.getElementById('last-updated').textContent = new Date().toLocaleString();
                            })
                            .catch(error => console.error('Error fetching status:', error));
                    }

                    function updateServices(services) {
                        const container = document.getElementById('services-container');
                        container.innerHTML = '';

                        for (const [name, service] of Object.entries(services)) {
                            const statusClass = `status-${service.status}`;
                            const uptime = service.uptime ? Math.round(service.uptime / 60) : 0;

                            container.innerHTML += `
                                <div class="service-card">
                                    <h3>${service.name} (Port ${service.port})</h3>
                                    <p>Status: <span class="${statusClass}">${service.status.toUpperCase()}</span></p>
                                    <p>Restarts: ${service.restart_count}</p>
                                    <p>Uptime: ${uptime} minutes</p>
                                    <div class="metrics">
                                        <div class="metric">
                                            <div class="metric-value">${service.cpu_percent.toFixed(1)}%</div>
                                            <div class="metric-label">CPU</div>
                                        </div>
                                        <div class="metric">
                                            <div class="metric-value">${service.memory_mb.toFixed(1)}</div>
                                            <div class="metric-label">Memory (MB)</div>
                                        </div>
                                        <div class="metric">
                                            <div class="metric-value">${service.memory_percent.toFixed(1)}%</div>
                                            <div class="metric-label">Memory %</div>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }
                    }

                    function updateAlerts(alerts) {
                        const container = document.getElementById('alerts-container');
                        container.innerHTML = '';

                        if (alerts.length === 0) {
                            container.innerHTML = '<p>No recent alerts</p>';
                            return;
                        }

                        alerts.forEach(alert => {
                            const alertClass = `alert-${alert.severity}`;
                            const time = new Date(alert.timestamp).toLocaleString();

                            container.innerHTML += `
                                <div class="alert ${alertClass}">
                                    <strong>${alert.service_name}</strong> - ${alert.event_type}<br>
                                    <small>${time}</small><br>
                                    ${alert.message}
                                </div>
                            `;
                        });
                    }

                    function updateSystemInfo(systemInfo) {
                        const container = document.getElementById('system-info-container');
                        container.innerHTML = `
                            <div class="metrics">
                                <div class="metric">
                                    <div class="metric-value">${systemInfo.cpu_percent.toFixed(1)}%</div>
                                    <div class="metric-label">System CPU</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">${systemInfo.memory_percent.toFixed(1)}%</div>
                                    <div class="metric-label">System Memory</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">${systemInfo.memory_total.toFixed(1)} GB</div>
                                    <div class="metric-label">Total Memory</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">${systemInfo.disk_usage.toFixed(1)}%</div>
                                    <div class="metric-label">Disk Usage</div>
                                </div>
                            </div>
                        `;
                    }

                    // Initial load and refresh every 10 seconds
                    getStatus();
                    setInterval(getStatus, 10000);
                </script>
            </body>
            </html>
            """

            @app.route('/')
            def dashboard():
                return render_template_string(HTML_TEMPLATE)

            @app.route('/api/status')
            def api_status():
                return jsonify(self.get_status())

            @app.route('/api/start/<service_name>')
            def api_start_service(service_name):
                if service_name in self.services:
                    success = self.start_service(self.services[service_name])
                    return jsonify({"success": success})
                return jsonify({"success": False, "error": "Service not found"})

            @app.route('/api/stop/<service_name>')
            def api_stop_service(service_name):
                if service_name in self.services:
                    success = self.stop_service(self.services[service_name])
                    return jsonify({"success": success})
                return jsonify({"success": False, "error": "Service not found"})

            @app.route('/api/restart/<service_name>')
            def api_restart_service(service_name):
                if service_name in self.services:
                    success = self.restart_service(self.services[service_name])
                    return jsonify({"success": success})
                return jsonify({"success": False, "error": "Service not found"})

            port = self.config.get("web_dashboard", {}).get("port", 8080)
            app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

        except ImportError:
            self.logger.warning("Flask not installed, web dashboard disabled")
        except Exception as e:
            self.logger.error(f"Error starting web dashboard: {e}")

def main():
    """Main entry point"""
    monitor = ServiceMonitor()

    try:
        monitor.start()

        # Keep the main thread alive
        while monitor.running:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()