import os
import logging

class CustomLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathname = custom_path_filter(self.pathname)

def custom_path_filter(pathname):
    project_root = "react-trip-planner"

    if project_root in pathname:
        return pathname.split(project_root)[1]
    else:
        return pathname
    
def setup_logging(log_filename = "app.log", log_dir = "logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_filename)

    logging.setLogRecordFactory(CustomLogRecord)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )

    return logging.getLogger(__name__)

logger = setup_logging()
    