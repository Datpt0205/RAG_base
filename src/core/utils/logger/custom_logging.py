import logging
from typing import List

from src.core.utils.logger.handlers import Handlers
from src.core.utils.config import settings


class LogHandler(object):

    def __init__(self):
        self.available_handlers: List = Handlers().get_handlers()

    def get_logger(self, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(settings.LOG_LEVEL)
        if logger.hasHandlers():
            logger.handlers.clear()
        for handler in self.available_handlers:
            logger.addHandler(handler)
        logger.propagate = False
        return logger


class LoggerMixin:
    def __init__(self) -> None:
        log_handler = LogHandler()
        self.logger = log_handler.get_logger(__name__)
        self.logger.setLevel(settings.LOG_LEVEL)
