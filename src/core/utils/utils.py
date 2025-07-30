import re
from datetime import datetime
from src.app import logger_instance

logger = logger_instance.get_logger(__name__)
UUID_PATTERN = re.compile("^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

extension_mapping = {
        'pdf': 'pdf',
        'docx': 'word',
        'doc': 'word',  
        'pptx': 'pptx',
        'ppt': 'pptx',
        'png': 'image',  
        'jpg': 'image',  
        'jpeg': 'image',  
        'bmp': 'image'
    }

def get_current_timestamp_string():
    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp_str