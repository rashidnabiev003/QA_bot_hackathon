import pytest
from unittest.mock import Mock, patch
from src.retriever.metrics import (
    _tok_ru,
    _stem_ru,
    rouge_l_f1_ru,
    ndcg_at_k,
    mrr_at_k,
    map_at_100,
    MetricComputer,
)


class TestTokenization:
    def test_tok_ru_basic(self):
        assert _tok_ru("Привет мир") == ["привет", "мир"]
        
    def test_tok_ru_empty(self):
        assert _tok_ru("") == []
        
    def test_tok_ru_with_punctuation(self):
        tokens = _tok_ru("Привет, как дела?")
        assert "привет" in tokens
        assert "как" in tokens
        assert "дела" in tokens


class TestRougeL:
    def test_rouge_l_identical_strings(self):
        score = rouge_l_f1_ru("привет мир", "привет мир")
        assert score > 0.9
        
    def test_rouge_l_no_overlap(self):
        score = rouge_l_f1_ru("привет", "пока")
        assert score == 0.0
        
    def test_rouge_l_partial_overlap(self):
        score = rouge_l_f1_ru("привет мир", "привет всем")
        assert 0 < score < 1


class TestRankingMetrics:
    def test_ndcg_first_position(self):
        assert ndcg_at_k(0, 10) == 1.0
        
    def test_ndcg_out_of_range(self):
        assert ndcg_at_k(10, 10) == 0.0
        assert ndcg_at_k(-1, 10) == 0.0
        
    def test_mrr_first_position(self):
        assert mrr_at_k(0, 10) == 1.0
        
    def test_mrr_second_position(self):
        assert mrr_at_k(1, 10) == 0.5
        
    def test_map_basic(self):
        assert map_at_100(0) == 1.0
        assert map_at_100(9) == 0.1


class TestMetricComputer:
    @pytest.fixture
    def metric_computer(self, mock_metric_config, mock_flag_reranker):
        with patch('src.retriever.metrics.FlagReranker', return_value=mock_flag_reranker):
            mc = MetricComputer(mock_metric_config)
            mc.sas = mock_flag_reranker
            return mc
    
    def test_rouge_l_computation(self, metric_computer):
        score = metric_computer.rougeL_ru("привет мир", "привет мир")
        assert score > 0.9
    
    def test_sas_computation(self, metric_computer):
        score = metric_computer.sas_user_bge_m3("test", "test")
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_retrieval_metrics_semantic(self, metric_computer):
        gold_ctx = "золотой контекст"
        retrieved = ["золотой контекст", "другой текст", "еще текст"]
        
        metrics = metric_computer.retrieval_metrics_semantic(
            gold_ctx, retrieved, k=3, rel_threshold=0.5
        )
        
        assert "ndcg" in metrics
        assert "mrr" in metrics
        assert "map" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())
    
    def test_retrieval_metrics_empty(self, metric_computer):
        metrics = metric_computer.retrieval_metrics_semantic("test", [], k=10)
        assert metrics["ndcg"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["map"] == 0.0
    
    def test_compute_all_with_gold(self, metric_computer):
        result = metric_computer.compute_all(
            query="вопрос",
            answer="ответ модели",
            contexts=["контекст 1", "контекст 2"],
            gold_answer="золотой ответ",
            gold_context="золотой контекст",
        )
        
        assert "bleurt" in result
        assert "sas" in result
        assert "rouge_l" in result
        assert "ndcg" in result
        assert "mrr" in result
        assert "map" in result
        
        for key, value in result.items():
            assert isinstance(value, float)
    
    def test_compute_all_without_gold(self, metric_computer):
        result = metric_computer.compute_all(
            query="вопрос",
            answer="ответ модели",
            contexts=["контекст 1", "контекст 2"],
        )
        
        assert all(v == 0.0 for v in result.values())


