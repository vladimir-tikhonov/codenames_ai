import configparser


def read_app_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config
