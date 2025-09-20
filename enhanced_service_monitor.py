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
from typing import Dict, List, Optional, Tuple, Any
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
import traceback

# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# Try to import rich for enhanced terminal dashboard
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.console import Group
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class ServiceStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    CRASHED = "crashed"
    RESTARTING = "restarting"
    UNKNOWN = "unknown"
    STARTING = "starting"

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
    status_check_url: str = ""
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
    dependencies: List[str] = None

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
    load_average: List[float]

class EnhancedServiceMonitor:
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
        self.command_queue = deque()
        self.service_logs = {}

        # Network stats tracking
        self.last_network_stats = None

        # Initialize services
        self._initialize_services()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("enhanced_service_monitor")
        logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_dir / f"service_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
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
                    "restart_cooldown": 60,
                    "dependencies": []
                },
                "frontend": {
                    "port": 5173,
                    "health_check_url": "http://localhost:5173",
                    "start_command": "npm run dev",
                    "working_dir": "ai-game-assistant",
                    "auto_restart": True,
                    "max_restarts": 5,
                    "restart_cooldown": 60,
                    "dependencies": ["backend"]
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
                status_check_url=service_config.get("status_check_url", ""),
                start_command=service_config.get("start_command", ""),
                working_dir=service_config.get("working_dir", ""),
                auto_restart=service_config.get("auto_restart", True),
                max_restarts=service_config.get("max_restarts", 5),
                restart_cooldown=service_config.get("restart_cooldown", 60),
                dependencies=service_config.get("dependencies", [])
            )
            self.services[service_name] = service
            self.service_logs[service_name] = deque(maxlen=1000)
            self.logger.info(f"Initialized service: {service_name}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def add_alert(self, service_name: str, event_type: str, message: str, severity: str = "info", details: Dict = None):
        """Add an alert event"""
        alert = AlertEvent(
            timestamp=datetime.now(),
            service_name=service_name,
            event_type=event_type,
            message=message,
            severity=severity,
            details=details or {}
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

            # Load average (Unix-like systems)
            load_average = []
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                load_average = [0.0, 0.0, 0.0]

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
                process_count=process_count,
                load_average=load_average
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

    def start_service(self, service: ServiceInfo) -> bool:
        """Start a service"""
        try:
            if service.status == ServiceStatus.RUNNING:
                self.logger.info(f"Service {service.name} is already running")
                return True

            self.add_alert(service.name, "START_ATTEMPT", f"Starting service {service.name}")

            # Check dependencies
            if service.dependencies:
                for dep_name in service.dependencies:
                    if dep_name in self.services:
                        dep_service = self.services[dep_name]
                        if dep_service.status != ServiceStatus.RUNNING:
                            self.add_alert(service.name, "DEPENDENCY_ERROR",
                                         f"Dependency {dep_name} is not running", "warning")
                            return False

            # Change to working directory
            old_cwd = os.getcwd()
            work_dir = Path.cwd() / service.working_dir

            if not work_dir.exists():
                self.add_alert(service.name, "START_ERROR",
                             f"Working directory does not exist: {work_dir}", "error")
                return False

            os.chdir(work_dir)

            # Start the process
            service.status = ServiceStatus.STARTING
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
                service.error_count = 0
                service.last_error = ""
                self.add_alert(service.name, "START_SUCCESS",
                             f"Service {service.name} started successfully", "info")
                return True
            else:
                service.status = ServiceStatus.CRASHED
                service.error_count += 1
                service.last_error = "Service failed to start or respond to health checks"
                self.add_alert(service.name, "START_FAILED",
                             f"Service {service.name} failed to start", "error")
                return False

        except Exception as e:
            service.status = ServiceStatus.CRASHED
            service.error_count += 1
            service.last_error = str(e)
            self.add_alert(service.name, "START_ERROR",
                         f"Error starting service {service.name}: {e}", "error")
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
        if service.restart_count >= service.max_restarts:
            time_since_last_restart = (datetime.now() - service.last_restart).total_seconds() if service.last_restart else 0
            if time_since_last_restart < service.restart_cooldown:
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
        self.logger.info("Starting enhanced service monitoring...")

        last_resource_check = 0

        while self.running:
            try:
                current_time = time.time()

                # Service health checks
                for service in self.services.values():
                    old_status = service.status
                    is_healthy = self.check_service_health(service)

                    if is_healthy:
                        if service.status != ServiceStatus.RUNNING:
                            service.status = ServiceStatus.RUNNING
                            service.last_restart = datetime.now()
                            self.add_alert(service.name, "HEALTHY",
                                         f"Service {service.name} is healthy", "info")

                        # Calculate uptime
                        if service.last_restart:
                            service.uptime = (datetime.now() - service.last_restart).total_seconds()
                    else:
                        if service.status == ServiceStatus.RUNNING:
                            service.status = ServiceStatus.CRASHED
                            service.error_count += 1
                            self.add_alert(service.name, "UNHEALTHY",
                                         f"Service {service.name} is unhealthy", "warning")

                            # Try to restart the service if auto-restart is enabled
                            if service.auto_restart:
                                if self.restart_service(service):
                                    self.add_alert(service.name, "RESTART_SUCCESS",
                                                 f"Service {service.name} restarted successfully", "info")
                                else:
                                    self.add_alert(service.name, "RESTART_FAILED",
                                                 f"Failed to restart service {service.name}", "error")

                    service.last_check = datetime.now()

                    # Get resource usage
                    cpu_percent, memory_percent, memory_mb = self.get_process_resources(service)
                    service.cpu_percent = cpu_percent
                    service.memory_percent = memory_percent
                    service.memory_mb = memory_mb

                    # Check resource thresholds
                    if cpu_percent > self.cpu_threshold:
                        self.add_alert(service.name, "HIGH_CPU",
                                     f"High CPU usage: {cpu_percent:.1f}%", "warning")

                    if memory_percent > self.memory_threshold:
                        self.add_alert(service.name, "HIGH_MEMORY",
                                     f"High memory usage: {memory_percent:.1f}%", "warning")

                # System metrics collection
                if current_time - last_resource_check >= self.resource_check_interval:
                    metrics = self.get_system_metrics()
                    if metrics:
                        self.system_metrics_history.append(metrics)
                    last_resource_check = current_time

                # Process commands
                self._process_commands()

                # Sleep for the health check interval
                time.sleep(self.health_check_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.logger.debug(traceback.format_exc())
                time.sleep(5)

    def _process_commands(self):
        """Process commands from the command queue"""
        while self.command_queue:
            try:
                command = self.command_queue.popleft()
                self._execute_command(command)
            except Exception as e:
                self.logger.error(f"Error processing command: {e}")

    def _execute_command(self, command: Dict):
        """Execute a single command"""
        cmd_type = command.get("type")
        service_name = command.get("service")

        if service_name not in self.services:
            return

        service = self.services[service_name]

        if cmd_type == "start":
            self.start_service(service)
        elif cmd_type == "stop":
            self.stop_service(service)
        elif cmd_type == "restart":
            self.restart_service(service)

    def send_command(self, cmd_type: str, service_name: str):
        """Send a command to the monitor"""
        self.command_queue.append({
            "type": cmd_type,
            "service": service_name,
            "timestamp": datetime.now().isoformat()
        })

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
                    "response_time": service.response_time,
                    "uptime": service.uptime,
                    "error_count": service.error_count,
                    "last_error": service.last_error
                }
                for name, service in self.services.items()
            },
            "alerts": [asdict(alert) for alert in self.alerts[-10:]],  # Last 10 alerts
            "system_metrics": asdict(self.system_metrics_history[-1]) if self.system_metrics_history else None,
            "system_info": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                "memory_available": psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
                "disk_usage": psutil.disk_usage('/').percent,
                "uptime": time.time() - psutil.boot_time()
            }
        }

    def start_terminal_dashboard(self):
        """Start the terminal dashboard"""
        if not RICH_AVAILABLE:
            self.logger.warning("Rich library not available. Install with: pip install rich")
            return

        console = Console()

        def create_layout():
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="logs", size=8)
            )
            layout["main"].split_row(
                Layout(name="services"),
                Layout(name="system")
            )
            return layout

        def create_services_table():
            table = Table(title="Service Status", show_header=True, header_style="bold magenta")
            table.add_column("Service", style="cyan", no_wrap=True)
            table.add_column("Port", style="blue")
            table.add_column("Status", style="green")
            table.add_column("CPU %", style="yellow")
            table.add_column("Memory %", style="yellow")
            table.add_column("Memory MB", style="yellow")
            table.add_column("Response Time", style="blue")
            table.add_column("Restarts", style="red")
            table.add_column("Uptime", style="green")

            for service in self.services.values():
                status_text = service.status.value.upper()
                status_color = {
                    "RUNNING": "green",
                    "STOPPED": "red",
                    "CRASHED": "red",
                    "RESTARTING": "yellow",
                    "STARTING": "yellow",
                    "UNKNOWN": "gray"
                }.get(service.status.value, "white")

                response_text = f"{service.response_time:.1f}ms" if service.response_time > 0 else "N/A"
                uptime_text = f"{int(service.uptime / 60)}m" if service.uptime > 0 else "N/A"

                table.add_row(
                    service.name,
                    str(service.port),
                    Text(status_text, style=status_color),
                    f"{service.cpu_percent:.1f}%",
                    f"{service.memory_percent:.1f}%",
                    f"{service.memory_mb:.1f}",
                    response_text,
                    str(service.restart_count),
                    uptime_text
                )

            return table

        def create_system_panel():
            if not self.system_metrics_history:
                return Panel("No system data", title="System Metrics")

            metrics = self.system_metrics_history[-1]

            # Create progress bars
            cpu_bar = f"CPU: {metrics.cpu_total:.1f}% [{'█' * int(metrics.cpu_total / 5)}{' ' * (20 - int(metrics.cpu_total / 5))}]"
            memory_bar = f"Memory: {metrics.memory_percent:.1f}% [{'█' * int(metrics.memory_percent / 5)}{' ' * (20 - int(metrics.memory_percent / 5))}]"
            disk_bar = f"Disk: {metrics.disk_percent:.1f}% [{'█' * int(metrics.disk_percent / 5)}{' ' * (20 - int(metrics.disk_percent / 5))}]"

            system_info = f"""
System Uptime: {int(metrics.uptime / 3600)}h {int((metrics.uptime % 3600) / 60)}m
Active Processes: {metrics.process_count}
Load Average: {', '.join(f'{x:.2f}' for x in metrics.load_average)}

{cpu_bar}
{memory_bar}
{disk_bar}

Memory: {metrics.memory_used:.1f}GB / {metrics.memory_total:.1f}GB
Network: ↑{metrics.network_sent:.1f}MB ↓{metrics.network_received:.1f}MB
            """.strip()

            return Panel(system_info, title="System Metrics", border_style="blue")

        def create_logs_panel():
            recent_alerts = self.alerts[-8:] if self.alerts else []
            if not recent_alerts:
                return Panel("No recent alerts", title="Recent Activity")

            log_text = ""
            for alert in recent_alerts:
                color = {
                    'info': 'green',
                    'warning': 'yellow',
                    'error': 'red',
                    'critical': 'bold red'
                }.get(alert.severity, 'white')

                log_text += f"[{color}]{alert.timestamp.strftime('%H:%M:%S')} - {alert.service_name}: {alert.message}[/{color}]\n"

            return Panel(log_text.strip(), title="Recent Activity", border_style="green")

        def create_control_panel():
            controls = """
Service Controls:
• [S]tart service
• [T]op service
• [R]estart service
• [Q]uit monitor

Press a key to control services
            """.strip()
            return Panel(controls, title="Controls", border_style="yellow")

        layout = create_layout()

        with Live(layout, refresh_per_second=2, screen=True) as live:
            while self.running:
                try:
                    layout["header"].update(Panel(
                        f"🚀 Enhanced PyGB Service Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        style="bold white on blue"
                    ))

                    layout["services"].update(create_services_table())
                    layout["system"].update(create_system_panel())
                    layout["logs"].update(Group(create_logs_panel(), create_control_panel()))

                    time.sleep(0.5)

                except KeyboardInterrupt:
                    self.running = False
                    break

    def start_web_dashboard(self):
        """Start the web dashboard"""
        try:
            from flask import Flask, jsonify, render_template_string, request

            app = Flask(__name__)

            # Enhanced HTML template for the dashboard
            HTML_TEMPLATE = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Enhanced Service Monitor Dashboard</title>
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; }
                    .container { max-width: 1400px; margin: 0 auto; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                    .service-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 20px; }
                    .service-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #007bff; }
                    .service-card.crashed { border-left-color: #dc3545; }
                    .service-card.stopped { border-left-color: #6c757d; }
                    .service-card.restarting { border-left-color: #ffc107; }
                    .status { padding: 4px 8px; border-radius: 20px; font-size: 12px; font-weight: bold; text-transform: uppercase; }
                    .status.running { background-color: #d4edda; color: #155724; }
                    .status.stopped { background-color: #f8d7da; color: #721c24; }
                    .status.crashed { background-color: #f8d7da; color: #721c24; }
                    .status.restarting { background-color: #fff3cd; color: #856404; }
                    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-top: 15px; }
                    .metric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; text-align: center; }
                    .metric-value { font-size: 20px; font-weight: bold; color: #495057; }
                    .metric-label { font-size: 11px; color: #6c757d; margin-top: 4px; }
                    .system-panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
                    .system-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
                    .alerts-panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .alert { padding: 12px; margin-bottom: 10px; border-radius: 8px; border-left: 4px solid; }
                    .alert-info { background-color: #d1ecf1; border-color: #17a2b8; }
                    .alert-warning { background-color: #fff3cd; border-color: #ffc107; }
                    .alert-error { background-color: #f8d7da; border-color: #dc3545; }
                    .alert-critical { background-color: #f8d7da; border-color: #dc3545; }
                    .controls { margin-top: 15px; }
                    .btn { padding: 8px 16px; margin: 2px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
                    .btn-primary { background-color: #007bff; color: white; }
                    .btn-warning { background-color: #ffc107; color: black; }
                    .btn-danger { background-color: #dc3545; color: white; }
                    .btn-success { background-color: #28a745; color: white; }
                    .refresh-info { text-align: center; margin-top: 20px; color: #6c757d; font-size: 14px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 Enhanced Service Monitor Dashboard</h1>
                        <p>Real-time monitoring of backend and frontend services with comprehensive metrics</p>
                    </div>

                    <div id="services-container" class="service-grid"></div>

                    <div class="system-panel">
                        <h3>💻 System Information</h3>
                        <div id="system-info-container" class="system-grid"></div>
                    </div>

                    <div class="alerts-panel">
                        <h3>📊 Recent Alerts</h3>
                        <div id="alerts-container"></div>
                    </div>

                    <div class="refresh-info">
                        <p>🔄 Auto-refresh every 10 seconds | Last updated: <span id="last-updated">Never</span></p>
                    </div>
                </div>

                <script>
                    function getStatus() {
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {
                                updateServices(data.services);
                                updateSystemInfo(data.system_info, data.system_metrics);
                                updateAlerts(data.alerts);
                                document.getElementById('last-updated').textContent = new Date().toLocaleString();
                            })
                            .catch(error => console.error('Error fetching status:', error));
                    }

                    function updateServices(services) {
                        const container = document.getElementById('services-container');
                        container.innerHTML = '';

                        for (const [name, service] of Object.entries(services)) {
                            const statusClass = service.status.toLowerCase();
                            const uptime = service.uptime ? Math.round(service.uptime / 60) : 0;
                            const responseTime = service.response_time || 0;

                            container.innerHTML += `
                                <div class="service-card ${statusClass}">
                                    <h3>${service.name}</h3>
                                    <p>Port: ${service.port} | Status: <span class="status ${statusClass}">${service.status.toUpperCase()}</span></p>
                                    <p>Restarts: ${service.restart_count} | Errors: ${service.error_count}</p>
                                    <p>Uptime: ${uptime} minutes | Response: ${responseTime.toFixed(1)}ms</p>
                                    ${service.last_error ? `<p style="color: #dc3545; font-size: 12px;">Error: ${service.last_error}</p>` : ''}
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
                                    <div class="controls">
                                        <button class="btn btn-success" onclick="controlService('start', '${name}')">Start</button>
                                        <button class="btn btn-danger" onclick="controlService('stop', '${name}')">Stop</button>
                                        <button class="btn btn-warning" onclick="controlService('restart', '${name}')">Restart</button>
                                    </div>
                                </div>
                            `;
                        }
                    }

                    function updateSystemInfo(systemInfo, systemMetrics) {
                        const container = document.getElementById('system-info-container');

                        let metricsHtml = `
                            <div class="metric">
                                <div class="metric-value">${systemInfo.cpu_percent.toFixed(1)}%</div>
                                <div class="metric-label">CPU Usage</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${systemInfo.memory_percent.toFixed(1)}%</div>
                                <div class="metric-label">Memory Usage</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${systemInfo.memory_total.toFixed(1)} GB</div>
                                <div class="metric-label">Total Memory</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${systemInfo.disk_usage.toFixed(1)}%</div>
                                <div class="metric-label">Disk Usage</div>
                            </div>
                        `;

                        if (systemMetrics) {
                            metricsHtml += `
                                <div class="metric">
                                    <div class="metric-value">${systemMetrics.process_count}</div>
                                    <div class="metric-label">Processes</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">${Math.round(systemMetrics.uptime / 3600)}h</div>
                                    <div class="metric-label">Uptime</div>
                                </div>
                            `;
                        }

                        container.innerHTML = metricsHtml;
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

                    function controlService(action, serviceName) {
                        fetch(`/api/${action}/${serviceName}`, { method: 'POST' })
                            .then(response => response.json())
                            .then(data => {
                                if (data.success) {
                                    alert(`${action} command sent to ${serviceName}`);
                                } else {
                                    alert(`Failed to ${action} ${serviceName}: ${data.error}`);
                                }
                            })
                            .catch(error => {
                                alert(`Error controlling service: ${error}`);
                            });
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

            @app.route('/api/start/<service_name>', methods=['POST'])
            def api_start_service(service_name):
                if service_name in self.services:
                    self.send_command('start', service_name)
                    return jsonify({"success": True})
                return jsonify({"success": False, "error": "Service not found"})

            @app.route('/api/stop/<service_name>', methods=['POST'])
            def api_stop_service(service_name):
                if service_name in self.services:
                    self.send_command('stop', service_name)
                    return jsonify({"success": True})
                return jsonify({"success": False, "error": "Service not found"})

            @app.route('/api/restart/<service_name>', methods=['POST'])
            def api_restart_service(service_name):
                if service_name in self.services:
                    self.send_command('restart', service_name)
                    return jsonify({"success": True})
                return jsonify({"success": False, "error": "Service not found"})

            port = self.config.get("dashboards", {}).get("web", {}).get("port", 8080)
            app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

        except ImportError:
            self.logger.warning("Flask not installed, web dashboard disabled")
        except Exception as e:
            self.logger.error(f"Error starting web dashboard: {e}")

    def start(self):
        """Start the monitoring system"""
        if self.running:
            self.logger.warning("Monitor is already running")
            return

        self.running = True
        self.logger.info("Starting enhanced service monitoring system...")

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        self.monitor_thread.start()

        # Start web dashboard if enabled
        if self.enable_web_dashboard:
            self.web_server_thread = threading.Thread(target=self.start_web_dashboard, daemon=True)
            self.web_server_thread.start()

        # Start terminal dashboard if enabled
        if self.enable_terminal_dashboard:
            self.terminal_dashboard_thread = threading.Thread(target=self.start_terminal_dashboard, daemon=True)
            self.terminal_dashboard_thread.start()

        # Initial service start
        for service in self.services.values():
            self.start_service(service)

        self.logger.info("Enhanced service monitoring system started")

    def stop(self):
        """Stop the monitoring system"""
        if not self.running:
            return

        self.logger.info("Stopping enhanced service monitoring system...")
        self.running = False

        # Stop all services
        for service in self.services.values():
            self.stop_service(service)

        # Wait for threads to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        if self.web_server_thread:
            self.web_server_thread.join(timeout=5)

        if self.terminal_dashboard_thread:
            self.terminal_dashboard_thread.join(timeout=5)

        self.logger.info("Enhanced service monitoring system stopped")

def main():
    """Main entry point"""
    monitor = EnhancedServiceMonitor()

    try:
        monitor.start()

        # Keep the main thread alive if not using terminal dashboard
        if not monitor.enable_terminal_dashboard or not RICH_AVAILABLE:
            while monitor.running:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()