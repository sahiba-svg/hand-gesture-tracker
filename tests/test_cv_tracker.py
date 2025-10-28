import importlib


def test_import_and_version():
    mod = importlib.import_module('CV_TRACKER_PROJECT')
    assert hasattr(mod, '__version__')
    assert isinstance(mod.__version__, str)
