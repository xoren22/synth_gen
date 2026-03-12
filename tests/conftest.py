def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: performance/benchmark tests")
