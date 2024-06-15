# logger_config.py
from loguru import logger
from datetime import datetime

# logger config
fmt = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | pid=<cyan>{process}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
charging_station_config = {"num":10}
log_path = f"../log/agent_num-{charging_station_config['num']}/{datetime.now().strftime('%Y%m-%d_%H:%M:%S')}.log"
logger.add(log_path, format=fmt, rotation="10 MB", enqueue=True, backtrace=True)

# Export the logger to be used in other modules
configured_logger = logger
