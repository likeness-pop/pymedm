import ast

import pytest

# --------------------------------------------------------------
# adding command line options
# ---------------------------


def pytest_addoption(parser):
    """Add custom command line arguments to the testing suite"""

    # flag for local or remote/VM testing
    parser.addoption(
        "--local",
        action="store",
        default="True",
        help="Boolean flag for local or remote/VM testing.",
        choices=("True", "False"),
        type=str,
    )

    # flag for `dev` environment testing (bleeding edge dependencies)
    parser.addoption(
        "--env",
        action="store",
        default="latest",
        help=(
            "Environment type label of dependencies for determining whether certain "
            "tests should be run. Generally we are working with minimum/oldest, "
            "latest/stable, and bleeding edge."
        ),
        type=str,
    )

    # flag for determining package manager used in testing
    parser.addoption(
        "--package_manager",
        action="store",
        default="micromamba",
        help=(
            "Package manager label of dependencies for determining "
            "whether certain tests should be run."
        ),
        type=str,
    )


def pytest_configure(config):
    """Set session attributes."""

    # ``local`` from ``pytest_addoption()``
    pytest.LOCAL = ast.literal_eval(config.getoption("local"))

    # ``env`` from ``pytest_addoption()``
    pytest.ENV = config.getoption("env")
    valid_env_suffix = ["min", "latest", "dev"]
    assert pytest.ENV.split("_")[-1] in valid_env_suffix

    # ``package_manager`` from ``pytest_addoption()``
    pytest.PACKAGE_MANAGER = config.getoption("package_manager")
    valid_package_managers = ["conda", "micromamba"]
    assert pytest.PACKAGE_MANAGER in valid_package_managers
