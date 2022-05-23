def pytest_addoption(parser):
    parser.addoption(
        "--project_name",
        action="append",
        default=[],
    )


def pytest_generate_tests(metafunc):
    if "project_name" in metafunc.fixturenames:
        metafunc.parametrize("project_name", metafunc.config.getoption("project_name"))
