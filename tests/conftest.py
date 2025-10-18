import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_flag_reranker():
    mock = Mock()
    mock.compute_score = Mock(return_value=[0.8, 0.6, 0.4])
    return mock


@pytest.fixture
def mock_metric_config():
    from src.schemas.pydantic_schemas import MetricConfig
    return MetricConfig(
        bleurt_ckpt="mock-bleurt",
        sas_model="mock-sas",
        sas_device="cpu",
        sas_fp16=False,
        bleurt_endpoint="http://mock-bleurt:8000",
    )


@pytest.fixture
def mock_build_config():
    from src.schemas.pydantic_schemas import BuildConfig
    return BuildConfig(
        batch_size=16,
        force_cpu=True,
        rerank_device="cpu",
        use_fp16_rerank=False,
        chunk_size=128,
        overlap_size=50,
    )


