from pydantic import BaseModel

class BuildConfig(BaseModel):
	batch_size: int
	force_cpu: bool               
	rerank_device: str        
	use_fp16_rerank: bool
	chunk_size: int
	overlap_size: int
