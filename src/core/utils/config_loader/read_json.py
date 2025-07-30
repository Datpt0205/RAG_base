import json

from pathlib import Path

from src.core.utils.config import settings
from src.core.utils.config_loader.config_interface import ConfigReaderInterface


class JsonConfigReader(ConfigReaderInterface):

    def __init__(self):
        super(JsonConfigReader, self).__init__()

    def read_config_from_file(self, config_filename: str):
        conf_path = Path(__file__).joinpath(settings.APP_CONFIG.SETTINGS_DIR, config_filename)
        with open(conf_path) as file:
            config = json.load(file)
        return config
