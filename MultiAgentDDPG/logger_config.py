# logger_config.py
import sys
from loguru import logger
from datetime import datetime

# logger config
fmt = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | pid=<cyan>{process}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
charging_station_config = {"num": 10}
log_path = f"../log/agent_num-{charging_station_config['num']}/{datetime.now().strftime('%Y%m-%d_%H:%M:%S')}.log"

# add a console handler to print the log
def filter_console(record):
    return "console" in record["extra"] and record["extra"]["console"]

# remove the default logger
logger.remove()

# add a file handler to save the log
logger.add(log_path, format=fmt, rotation="10 MB", enqueue=True, backtrace=True)
logger.add(sys.stdout, format=fmt, filter=filter_console) # add a console handler to print the log

# Export the logger to be used in other modules
configured_logger = logger


# # 這個訊息只會存儲在檔案中
# logger.info(f'No EVs have arrived at {env.timestamp}')
# # 這個訊息會顯示在控制台並存儲在檔案中
# logger.bind(console=True).info('This message will be shown in the console and saved in the log file')