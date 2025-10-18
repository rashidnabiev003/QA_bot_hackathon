import pytest
from src.schemas.pydantic_schemas import (
    BuildConfig,
    MetricConfig,
    ChatRequest,
    BenchmarkRequest,
)


class TestBuildConfig:
    def test_valid_config(self):
        config = BuildConfig(
            batch_size=32,
            force_cpu=False,
            rerank_device="cuda",
            use_fp16_rerank=True,
            chunk_size=256,
            overlap_size=50,
        )
        assert config.batch_size == 32
        assert config.chunk_size == 256
        
    def test_default_values(self):
        config = BuildConfig(
            batch_size=16,
            force_cpu=True,
            rerank_device="cpu",
            chunk_size=128,
            overlap_size=50,
        )
        assert config.use_fp16_rerank is True
        assert config.enable_rerank is True


class TestMetricConfig:
    def test_valid_config(self):
        config = MetricConfig(
            bleurt_ckpt="BLEURT-20",
            sas_model="BAAI/bge-m3",
            sas_device="cpu",
            sas_fp16=False,
            bleurt_endpoint="http://localhost:8000",
        )
        assert config.sas_device == "cpu"
        assert config.sas_fp16 is False


class TestChatRequest:
    def test_basic_request(self):
        req = ChatRequest(query="Привет")
        assert req.query == "Привет"
        assert req.use_llm_rerank is None
        
    def test_with_llm_rerank(self):
        req = ChatRequest(query="Тест", use_llm_rerank=True)
        assert req.use_llm_rerank is True


class TestBenchmarkRequest:
    def test_default_values(self):
        req = BenchmarkRequest()
        assert req.path is None
        assert req.limit is None
        assert req.use_llm_rerank is None
        
    def test_with_values(self):
        req = BenchmarkRequest(
            path="/path/to/file.xlsx",
            limit=100,
            use_llm_rerank=True,
        )
        assert req.path == "/path/to/file.xlsx"
        assert req.limit == 100

