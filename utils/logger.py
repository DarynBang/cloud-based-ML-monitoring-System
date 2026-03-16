import logging
import sys

def setup_logger(name: str = "trading_system", level=logging.INFO):
    """
    Thiết lập logger chuẩn cho toàn bộ hệ thống.
    """
    logger = logging.getLogger(name)
    
    # Tắt propagate để không gửi log lên các logger cha (tránh double logging)
    logger.propagate = False
    
    # Tránh việc add handler nhiều lần nếu hàm này được gọi lại
    if not logger.handlers:
        logger.setLevel(level)
        
        # Định dạng log: Thời gian - Tên Module - Mức độ - Thông điệp
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

# Tạo một logger mặc định
logger = setup_logger()
