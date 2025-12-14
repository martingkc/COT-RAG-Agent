# COT-RAG-Agent

This project explores an agentic RAG setup by fine-tuning an LLM with UNSLOTH to act as a query-generating controller that iteratively retrieves and conditions on external documents from a vector database. The code is old approx 1,5 years old, and it might not function as expected. 

The repository contains the following notebooks:
	-	Inference (inference.ipynb): Connects to a vector database (ChromaDB) and runs an iterative retrieve-and-generate loop.
	-	Synthetic Data Generation (generate_dataset.ipynb): Generates a synthetic training dataset used for fine-tuning (make sure to comply with the terms of the model used to generate the data).
	-	Fine-tuning (cot_finetune.ipynb): LoRA fine-tuning using UNSLOTH and TRL.

## How does it work?

The model is fine-tuned on a dataset that enforces a structured output format using the following tags:
-	<think>: Intermediate planning and query-selection block.
-	<query>: Indicates a call to the vector DB search function; contains only the query string.
-	<query_res>: Placeholder where retrieved documents are injected after a search.
-	<results>: Final structured output containing the selected, relevant evidence.

During inference, the model generates blocks of approximately 1000 tokens. After each generation cycle, the output is scanned for structured tags. When a <query> tag is detected, the query is extracted, executed against the vector DB, and the resulting documents are injected into the prompt inside a <query_res> block. The updated prompt is then fed back into the model. This loop continues until a <results> tag is produced or a stopping condition is reached.

## Results and limitations

The approach works for a small number of tool calls (typically 2–3), but degrades as the context grows. The main observed limitations are:
-	Context growth: Injected retrieval results can be long, quickly exhausting the context window and reducing instruction-following reliability.
-	Training inference mismatch: The synthetic dataset includes full retrieval outputs, which encourages the model to treat <query_res> content as generative rather than externally provided, weakening grounding.
-	Weak inference loop: Prompt splicing can corrupt formatting (duplicate or truncated tags) when generations are cut mid-structure.

Together, these issues lead to malformed outputs, query drift, and weak grounding in retrieved evidence once the loop runs for multiple iterations.

This was a short exploratory project I did **1,5 years ago** aimed at understanding synthetic data generation, rag and finetuning. If you ever need a stable rag agent just use langchain. 

Model artifacts are available on Hugging Face:
Martingkc/llama_lora_merged_model_v3￼
Repositories containing llama_lora correspond to LoRA adapters; models with merged in their name are merged checkpoints. MLX versions are also provided but run slowly on an M1 MacBook Pro (16GB).

## TODO
-	Code cleanup and refactoring into a minimal, reproducible pipeline
