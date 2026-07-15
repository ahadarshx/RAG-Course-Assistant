import importlib
import os


def test_boolean_config_parsing():
    os.environ["STRICT_EXTRACTIVE"] = "false"
    import backend.config
    config = importlib.reload(backend.config)
    assert config.settings.strict_extractive is False
