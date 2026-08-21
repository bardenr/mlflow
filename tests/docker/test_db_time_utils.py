import os
import platform
import shutil
import subprocess
import time

import pytest
from sqlalchemy import Engine, create_engine, select
from testcontainers.mssql import SqlServerContainer
from testcontainers.mysql import MySqlContainer
from testcontainers.postgres import PostgresContainer

from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES
from mlflow.store.db.utils import get_current_time_millis_expression

SKIP_MSSQL = platform.machine() == "arm64"


@pytest.fixture(scope="module", autouse=True)
def _configure_testcontainers() -> None:

    podman = shutil.which("podman")
    if podman is None:
        yield
        return

    result = subprocess.run(
        [podman, "machine", "inspect", "--format", "{{.ConnectionInfo.PodmanSocket.Path}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    podman_socket = result.stdout.strip()
    assert podman_socket is not None
    assert podman_socket != ""

    ENV_DOCKER_HOST = "DOCKER_HOST"
    ENV_TESTCONTAINERS_RYUK_DISABLED = "TESTCONTAINERS_RYUK_DISABLED"

    docker_host = os.environ.get(ENV_DOCKER_HOST)
    ryuk_disabled = os.environ.get(ENV_TESTCONTAINERS_RYUK_DISABLED)

    if docker_host is None:
        os.environ[ENV_DOCKER_HOST] = f"unix://{podman_socket}"

    if ryuk_disabled is None:
        os.environ[ENV_TESTCONTAINERS_RYUK_DISABLED] = "true"

    yield

    if docker_host is None:
        _ = os.environ.pop(ENV_DOCKER_HOST, None)

    if ryuk_disabled is None:
        _ = os.environ.pop(ENV_TESTCONTAINERS_RYUK_DISABLED, None)


@pytest.fixture(scope="module")
def mssql_engine() -> Engine:

    if SKIP_MSSQL:
        pytest.skip("MSSQL test unavailable on arm64 platforms")

    with SqlServerContainer().with_kwargs(platform="linux/amd64") as container:
        yield create_engine(container.get_connection_url())


@pytest.fixture(scope="module")
def mysql_engine() -> Engine:

    with MySqlContainer() as container:
        url = container.get_connection_url().replace("mysql://", "mysql+pymysql://", 1)
        yield create_engine(url)


@pytest.fixture(scope="module")
def postgres_engine() -> Engine:

    with PostgresContainer() as container:
        yield create_engine(container.get_connection_url())


@pytest.mark.parametrize(
    ("db_type", "db_engine_fixture"),
    [
        (MSSQL, mssql_engine.__name__),
        (MYSQL, mysql_engine.__name__),
        (POSTGRES, postgres_engine.__name__),
    ],
    ids=[MSSQL, MYSQL, POSTGRES],
)
def test_get_current_time_millis_expression_millisecond_precision(
    request: pytest.FixtureRequest, db_type: str, db_engine_fixture: str
) -> None:

    db_engine: Engine = request.getfixturevalue(db_engine_fixture)
    db_now = get_current_time_millis_expression(db_type=db_type)

    with db_engine.connect() as connection:
        before = connection.execute(select(db_now)).scalar()

    time.sleep(0.001)
    with db_engine.connect() as connection:
        result = connection.execute(select(db_now)).scalar()

    time.sleep(0.001)
    with db_engine.connect() as connection:
        after = connection.execute(select(db_now)).scalar()

    assert before < result < after
