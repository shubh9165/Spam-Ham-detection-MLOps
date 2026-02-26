import os 
import logging
from datetime import datetime


log_file=f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"

log_file_path=os.path.join(os.getcwd(),"logs",log_file)
os.makedirs(log_file_path,exist_ok=True)

Final_LOG_FILE_PATH=os.path.join(log_file_path,log_file)

logging.basicConfig(
    filename=Final_LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


