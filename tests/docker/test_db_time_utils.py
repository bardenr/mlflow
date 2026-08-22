import os
import platform
import shutil
import subprocess
import time

import pytest
from sqlalchemy import Engine, create_engine, select
from testcontainers.community.mssql import SqlServerContainer
from testcontainers.community.mysql import MySqlContainer
from testcontainers.community.postgres import PostgresContainer

from mlflow.server.jobs.lock_manager import JobLockManager
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES
from mlflow.store.db.utils import get_current_time_millis_expression
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.tracking.dbmodels.models import SqlSchedulerLease

SKIP_MSSQL = platform.machine() == "arm64"


@pytest.fixture(scope="module", autouse=True)
def _configure_testcontainers() -> None:
    """
    Check for podman binary and configure podman socket if available. If you are
    running podman on linux without the podman machine, you will need to manually
    open a socket by running ``podman system service --time=0`` prior to starting
    the tests.
    """

    podman = shutil.which("podman")
    if podman is None:
        yield
        return

    result = subprocess.run(
        [podman, "machine", "inspect", "--format", "{{.ConnectionInfo.PodmanSocket.Path}}"],
        capture_output=True,
        text=True,
        check=False,
    )

    podman_socket = None
    if result.returncode == 0:
        podman_socket = result.stdout.strip()

    XDG_RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR")
    if podman_socket is None and XDG_RUNTIME_DIR is not None:
        podman_socket = f"{XDG_RUNTIME_DIR}/podman/podman.sock"

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

    dialect = "pymysql"
    command = "--log-bin-trust-function-creators=1"
    with MySqlContainer(dialect=dialect, command=command) as container:
        yield create_engine(container.get_connection_url())


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

    time.sleep(0.002)
    with db_engine.connect() as connection:
        result = connection.execute(select(db_now)).scalar()

    time.sleep(0.002)
    with db_engine.connect() as connection:
        after = connection.execute(select(db_now)).scalar()

    assert before < result < after


@pytest.mark.parametrize(
    ("db_type", "db_engine_fixture"),
    [
        (MSSQL, mssql_engine.__name__),
        (MYSQL, mysql_engine.__name__),
        (POSTGRES, postgres_engine.__name__),
    ],
    ids=[MSSQL, MYSQL, POSTGRES],
)
def test_job_lock_manager_smoke_test(
    request: pytest.FixtureRequest, db_type: str, db_engine_fixture: str
) -> None:

    db_engine: Engine = request.getfixturevalue(db_engine_fixture)
    connection_url = db_engine.url.render_as_string(hide_password=False)
    job_store = SqlAlchemyJobStore(connection_url)
    lock_mgr = JobLockManager(job_store)

    lease_key = "scheduler-lease"
    ttl = 100

    # 1. Replica A acquires the lease at.
    lease_a_1 = lock_mgr.acquire_scheduler_lease(lease_key, ttl_seconds=ttl)
    assert lease_a_1 is not None
    assert lease_a_1.lease_key == lease_key
    assert lease_a_1.ttl_seconds == ttl

    # 2. Replica A renews within TTL (sleep prevents sub millisecond renewal)
    time.sleep(0.002)
    lease_a_2 = lock_mgr.renew_scheduler_lease(lease_a_1, ttl_seconds=ttl)
    assert lease_a_2.lease_key == lease_key
    assert lease_a_2.ttl_seconds == ttl
    assert lease_a_2.ttl_seconds == lease_a_1.ttl_seconds
    assert lease_a_2.acquired_at != lease_a_1.acquired_at

    # 3. Replica B attempts acquires the lease and is denied.
    assert lock_mgr.acquire_scheduler_lease(lease_key, ttl_seconds=ttl) is None

    # 4. Replica A attempts renewal with its old lease and fails
    assert lock_mgr.renew_scheduler_lease(lease_a_1, ttl_seconds=ttl) is None

    # 6. Verify final DB state matches Replica B's lease.
    with lock_mgr._session_maker(read_only=True) as session:
        row = (
            session.query(SqlSchedulerLease).filter(SqlSchedulerLease.lease_key == lease_key).one()
        )
        assert row.lease_key == lease_a_2.lease_key
        assert row.acquired_at == lease_a_2.acquired_at
        assert row.ttl_seconds == lease_a_2.ttl_seconds
