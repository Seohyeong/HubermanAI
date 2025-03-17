import logging

def setup_logging(log_level=logging.DEBUG):
    logger = logging.getLogger('RAG')
    logger.setLevel(log_level)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    
    # Set level for handlers
    console_handler.setLevel(log_level)
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()