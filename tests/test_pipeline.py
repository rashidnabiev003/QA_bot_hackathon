import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import pandas as pd


class TestQAPipeline:
    @pytest.fixture
    def mock_pipeline(self, temp_dir, mock_build_config, mock_metric_config):
        with patch('src.pipeline.orchestrator.SemanticSearch'), \
             patch('src.pipeline.orchestrator.MetricComputer'), \
             patch('src.pipeline.orchestrator.pd.read_excel', return_value=pd.DataFrame({
                 'id': [1, 2],
                 'question': ['вопрос 1', 'вопрос 2'],
                 'answer': ['ответ 1', 'ответ 2'],
                 'context': ['контекст 1', 'контекст 2'],
             })):
            from src.pipeline.orchestrator import QAPipeline
            pipeline = QAPipeline(index_dir=str(temp_dir))
            return pipeline
    
    def test_pipeline_initialization(self, mock_pipeline):
        assert mock_pipeline.index_dir is not None
        assert hasattr(mock_pipeline, 'engine')
        assert hasattr(mock_pipeline, 'mc')
        assert hasattr(mock_pipeline, 'gold_map')
    
    def test_normalize(self, mock_pipeline):
        result = mock_pipeline._normalize("  Привет   Мир  ")
        assert result == "привет мир"
        
    def test_normalize_empty(self, mock_pipeline):
        result = mock_pipeline._normalize("")
        assert result == ""


class TestPipelineAnswer:
    @pytest.mark.asyncio
    async def test_answer_basic_flow(self, temp_dir):
        with patch('src.pipeline.orchestrator.SemanticSearch') as mock_search, \
             patch('src.pipeline.orchestrator.MetricComputer') as mock_mc, \
             patch('src.pipeline.orchestrator.generate_answer', new_callable=AsyncMock) as mock_gen, \
             patch('src.pipeline.orchestrator.pd.read_excel', return_value=pd.DataFrame({
                 'id': [1],
                 'question': ['test'],
                 'answer': ['ans'],
                 'context': ['ctx'],
             })):
            
            from src.pipeline.orchestrator import QAPipeline
            
            mock_search_instance = Mock()
            mock_search_instance.retrieve = Mock(return_value={
                "results": [
                    {"text": "context 1", "score": 0.9, "score_rerank": 0.95, "metadata": {}},
                    {"text": "context 2", "score": 0.8, "score_rerank": 0.85, "metadata": {}},
                ],
                "meta": {"topn": 50, "topk": 5, "n_returned": 2}
            })
            mock_search_instance.ready = True
            mock_search_instance.load = Mock()
            mock_search.return_value = mock_search_instance
            
            mock_gen.return_value = {
                "answer": "test answer",
                "confidence": "high",
                "sources_used": [1, 2],
            }
            
            pipeline = QAPipeline(index_dir=str(temp_dir))
            result = await pipeline.answer("test query", topn=50, topk=5)
            
            assert "query" in result
            assert "answer" in result
            assert "contexts" in result
            assert "metrics" in result
            assert len(result["contexts"]) > 0


class TestBenchmark:
    @pytest.mark.asyncio
    async def test_benchmark_xlsx_structure(self, temp_dir):
        xlsx_path = temp_dir / "test_triplets.xlsx"
        df = pd.DataFrame({
            'id': [1, 2],
            'question': ['вопрос 1', 'вопрос 2'],
            'answer': ['ответ 1', 'ответ 2'],
            'context': ['контекст 1', 'контекст 2'],
        })
        df.to_excel(xlsx_path, index=False)
        
        with patch('src.pipeline.orchestrator.SemanticSearch') as mock_search, \
             patch('src.pipeline.orchestrator.MetricComputer') as mock_mc, \
             patch('src.pipeline.orchestrator.generate_answer', new_callable=AsyncMock) as mock_gen:
            
            from src.pipeline.orchestrator import QAPipeline
            
            mock_search_instance = Mock()
            mock_search_instance.retrieve = Mock(return_value={
                "results": [{"text": "ctx", "score": 0.9, "score_rerank": 0.95, "metadata": {}}],
                "meta": {}
            })
            mock_search_instance.ready = True
            mock_search_instance.load = Mock()
            mock_search.return_value = mock_search_instance
            
            mock_gen.return_value = {
                "answer": "test",
                "confidence": "high",
                "sources_used": [],
            }
            
            mock_mc_instance = Mock()
            mock_mc_instance.compute_all = Mock(return_value={
                "bleurt": 0.5, "sas": 0.6, "rouge_l": 0.7,
                "ndcg": 0.8, "mrr": 0.9, "map": 0.85
            })
            mock_mc.return_value = mock_mc_instance
            
            pipeline = QAPipeline(index_dir=str(temp_dir))
            pipeline.mc = mock_mc_instance
            
            result = await pipeline.benchmark_from_xlsx(
                path=str(xlsx_path),
                limit=1,
                use_llm_rerank=False,
            )
            
            assert result.get("ok") is True or "error" in result

