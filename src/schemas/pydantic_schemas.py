from pydantic import BaseModel


class BuildConfig(BaseModel):
	batch_size: int
	force_cpu: bool
	rerank_device: str
	use_fp16_rerank: bool = True
	chunk_size: int
	overlap_size: int
	enable_rerank: bool = True


class MetricConfig(BaseModel):
	bleurt_ckpt: str
	sas_model: str
	sas_device: str
	sas_fp16: bool
	bleurt_endpoint: str


class ScoreRequest(BaseModel):
	references: list[str]
	candidates: list[str]


class ChatRequest(BaseModel):
	query: str
	use_llm_rerank: bool | None = None


class BenchmarkRequest(BaseModel):
	path: str | None = None
	limit: int | None = None
	use_llm_rerank: bool | None = None
