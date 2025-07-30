class ConfigReaderInterface:

    def __init__(self):
        super().__init__()

    def read_config_from_file(self, config_filename: str):
        raise NotImplementedError()
