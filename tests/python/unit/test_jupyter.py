from omniff.integrations.jupyter import load_ipython_extension


def test_jupyter_extension_callable():
    assert callable(load_ipython_extension)
