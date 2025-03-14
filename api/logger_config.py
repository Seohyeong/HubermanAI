import logging
import os
from pathlib import Path
from datetime import datetime

def setup_logging(log_level=logging.INFO):
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger('RAG')
    logger.setLevel(log_level)
    
    # Create handlers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = str(Path(__file__).resolve().parent.parent)
    file_handler = logging.FileHandler(os.path.join(project_dir, f"logs/rag_{timestamp}.log"))
    console_handler = logging.StreamHandler()
    
    # Set level for handlers
    file_handler.setLevel(log_level)
    console_handler.setLevel(log_level)
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()