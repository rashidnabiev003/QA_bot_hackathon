import tkinter as tk
from tkinter import ttk, scrolledtext
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from src.retriever.semantic_search import SemanticSearch
from src.llm.llm_chat_bot_ollama import generate_answer
from src.llm.llm_reranker_vllm import rerank_with_llm
from src.retriever.metrics import MetricComputer
from src.schemas.pydantic_schemas import BuildConfig, MetricConfig

STYLE = {
	"font_family": "Arial",
	"font_size": 11,
	"bg": "#ffffff",
	"fg": "#000000",
	"input_bg": "#f0f0f0",
	"send_btn_color": "#10a37f",
	"panel_bg": "#f9f9f9",
	"border_color": "#e0e0e0",
}

class QABotApp:
	def __init__(self, root):
		self.root = root
		self.root.title("QA Bot с RAG")
		self.root.geometry("1400x800")
		self.root.minsize(1200, 600)
		
		self.is_processing = False
		self.use_llm_reranking = tk.BooleanVar(value=False)
		
		self.build_config = BuildConfig(
			batch_size=32,
			force_cpu=False,
			rerank_device="cuda",
			use_fp16_rerank=True,
			chunk_size=500,
			overlap_size=50
		)
		
		self.metric_config = MetricConfig(
			bleurt_ckpt="BLEURT-20",
			sas_model="BAAI/bge-m3",
			sas_device="cuda",
			sas_fp16=True,
			bleurt_endpoint=os.getenv("BLEURT_URL", "http://localhost:8088")
		)
		
		try:
			self.search_engine = SemanticSearch("./index", self.build_config)
			self.search_engine.load()
			self.metric_computer = MetricComputer(self.metric_config)
		except Exception as e:
			print(f"Ошибка инициализации: {e}")
			self.search_engine = None
			self.metric_computer = None
		
		self.create_widgets()
		self.apply_style()
	
	def create_widgets(self):
		main_container = tk.PanedWindow(
			self.root,
			orient=tk.HORIZONTAL,
			sashwidth=6,
			sashrelief=tk.FLAT,
			bg=STYLE["border_color"]
		)
		main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
		
		left_frame = tk.Frame(main_container, bg=STYLE["bg"])
		right_frame = tk.Frame(main_container, bg=STYLE["bg"])
		
		main_container.add(left_frame, minsize=500, width=700)
		main_container.add(right_frame, minsize=500, width=700)
		
		chat_label = tk.Label(
			left_frame,
			text="Чат",
			font=(STYLE["font_family"], 12, "bold"),
			bg=STYLE["bg"],
			fg=STYLE["fg"]
		)
		chat_label.pack(pady=(5, 0), anchor="w", padx=10)
		
		self.chat_text = scrolledtext.ScrolledText(
			left_frame,
			wrap=tk.WORD,
			font=(STYLE["font_family"], STYLE["font_size"]),
			bg=STYLE["bg"],
			fg=STYLE["fg"],
			padx=10,
			pady=10,
			relief=tk.FLAT,
			highlightthickness=1,
			highlightbackground=STYLE["border_color"]
		)
		self.chat_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
		self.chat_text.config(state=tk.DISABLED)
		
		input_container = tk.Frame(left_frame, bg=STYLE["bg"])
		input_container.pack(fill=tk.X, padx=10, pady=(0, 10))
		
		self.input_text = tk.Text(
			input_container,
			height=2,
			font=(STYLE["font_family"], STYLE["font_size"]),
			bg=STYLE["input_bg"],
			fg=STYLE["fg"],
			relief=tk.FLAT,
			padx=10,
			pady=8,
			wrap=tk.WORD,
			highlightthickness=1,
			highlightbackground=STYLE["border_color"]
		)
		self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
		self.input_text.bind("<Return>", self.on_enter)
		
		self.send_btn = tk.Button(
			input_container,
			text="→",
			font=(STYLE["font_family"], 16, "bold"),
			bg=STYLE["send_btn_color"],
			fg="white",
			relief=tk.FLAT,
			width=3,
			command=self.on_send
		)
		self.send_btn.pack(side=tk.RIGHT, padx=(5, 0))
		
		right_paned = tk.PanedWindow(
			right_frame,
			orient=tk.VERTICAL,
			sashwidth=6,
			sashrelief=tk.FLAT,
			bg=STYLE["border_color"]
		)
		right_paned.pack(fill=tk.BOTH, expand=True)
		
		context_frame = tk.Frame(right_paned, bg=STYLE["bg"])
		metrics_frame = tk.Frame(right_paned, bg=STYLE["bg"])
		
		right_paned.add(context_frame, minsize=200, height=400)
		right_paned.add(metrics_frame, minsize=150, height=400)
		
		ctx_header = tk.Frame(context_frame, bg=STYLE["bg"])
		ctx_header.pack(fill=tk.X, padx=10, pady=(5, 0))
		
		ctx_label = tk.Label(
			ctx_header,
			text="Найденные контексты",
			font=(STYLE["font_family"], 12, "bold"),
			bg=STYLE["bg"],
			fg=STYLE["fg"]
		)
		ctx_label.pack(side=tk.LEFT)
		
		self.llm_rerank_check = tk.Checkbutton(
			ctx_header,
			text="LLM реранкинг",
			variable=self.use_llm_reranking,
			font=(STYLE["font_family"], 10),
			bg=STYLE["bg"],
			fg=STYLE["fg"],
			selectcolor=STYLE["input_bg"]
		)
		self.llm_rerank_check.pack(side=tk.RIGHT)
		
		self.context_text = scrolledtext.ScrolledText(
			context_frame,
			wrap=tk.WORD,
			font=(STYLE["font_family"], STYLE["font_size"]-1),
			bg=STYLE["panel_bg"],
			fg="#333333",
			padx=10,
			pady=10,
			relief=tk.FLAT,
			highlightthickness=1,
			highlightbackground=STYLE["border_color"]
		)
		self.context_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
		self.context_text.config(state=tk.DISABLED)
		
		metrics_label = tk.Label(
			metrics_frame,
			text="Метрики",
			font=(STYLE["font_family"], 12, "bold"),
			bg=STYLE["bg"],
			fg=STYLE["fg"]
		)
		metrics_label.pack(pady=(5, 0), anchor="w", padx=10)
		
		self.metrics_text = scrolledtext.ScrolledText(
			metrics_frame,
			wrap=tk.WORD,
			font=(STYLE["font_family"], STYLE["font_size"]-1),
			bg=STYLE["panel_bg"],
			fg="#333333",
			padx=10,
			pady=10,
			relief=tk.FLAT,
			highlightthickness=1,
			highlightbackground=STYLE["border_color"]
		)
		self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
		self.metrics_text.config(state=tk.DISABLED)
	
	def apply_style(self):
		self.root.configure(bg=STYLE["bg"])
	
	def add_chat_message(self, text: str, is_user: bool = True):
		self.chat_text.config(state=tk.NORMAL)
		role = "Вы: " if is_user else "AI: "
		self.chat_text.insert(tk.END, f"{role}{text}\n\n")
		self.chat_text.config(state=tk.DISABLED)
		self.chat_text.see(tk.END)
	
	def update_contexts(self, contexts: List[Dict[str, Any]]):
		self.context_text.config(state=tk.NORMAL)
		self.context_text.delete(1.0, tk.END)
		
		for i, ctx in enumerate(contexts, 1):
			score = ctx.get('llm_score', ctx.get('score', 0))
			self.context_text.insert(tk.END, f"[{i}] Score: {score:.3f}\n", "header")
			self.context_text.insert(tk.END, f"{ctx['text']}\n\n")
		
		self.context_text.tag_config("header", font=(STYLE["font_family"], STYLE["font_size"], "bold"))
		self.context_text.config(state=tk.DISABLED)
	
	def update_metrics(self, metrics: Dict[str, Any]):
		self.metrics_text.config(state=tk.NORMAL)
		self.metrics_text.delete(1.0, tk.END)
		
		if metrics:
			for key, value in metrics.items():
				if isinstance(value, dict):
					self.metrics_text.insert(tk.END, f"{key}:\n")
					for k, v in value.items():
						self.metrics_text.insert(tk.END, f"  {k}: {v:.4f}\n")
				elif isinstance(value, (int, float)):
					self.metrics_text.insert(tk.END, f"{key}: {value:.4f}\n")
				else:
					self.metrics_text.insert(tk.END, f"{key}: {value}\n")
		
		self.metrics_text.config(state=tk.DISABLED)
	
	def on_enter(self, event):
		if event.state & 0x1:
			return
		self.on_send()
		return "break"
	
	def on_send(self):
		if self.is_processing:
			return
		
		query = self.input_text.get("1.0", tk.END).strip()
		if not query:
			return
		
		self.input_text.delete("1.0", tk.END)
		self.add_chat_message(query, is_user=True)
		self.is_processing = True
		self.send_btn.config(state=tk.DISABLED)
		
		asyncio.run(self.process_query(query))
		
		self.is_processing = False
		self.send_btn.config(state=tk.NORMAL)
	
	async def process_query(self, query: str):
		try:
			if not self.search_engine:
				self.add_chat_message("Система поиска не инициализирована", is_user=False)
				return
			
			result = self.search_engine.retrieve(query, topn=50, topk=10)
			contexts = result["results"]
			
			if self.use_llm_reranking.get():
				contexts = await rerank_with_llm(
					query=query,
					chunks=contexts,
					topk=5,
					vllm_url="http://localhost:8000"
				)
			
			self.update_contexts(contexts[:5])
			
			answer_data = await generate_answer(
				query=query,
				contexts=contexts[:5],
				ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
				model=os.getenv("OLLAMA_MODEL", "open-ai/gpt-oss-20b")
			)
			
			answer = answer_data.get("answer", "Ошибка генерации ответа")
			self.add_chat_message(answer, is_user=False)
			
			if self.metric_computer:
				try:
					metrics = {
						"confidence": answer_data.get("confidence", "unknown"),
						"sources_used": len(answer_data.get("sources_used", [])),
						"contexts_found": len(contexts)
					}
					self.update_metrics(metrics)
				except Exception as e:
					print(f"Ошибка вычисления метрик: {e}")
			
		except Exception as e:
			self.add_chat_message(f"Ошибка: {str(e)}", is_user=False)
			print(f"Ошибка обработки запроса: {e}")

if __name__ == "__main__":
	root = tk.Tk()
	app = QABotApp(root)
	root.mainloop()
