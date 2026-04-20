# HorizonBench

**Long-Horizon Personalization with Evolving Preferences**

User preferences evolve across months of interaction, and tracking them requires inferring when a stated preference has been changed by a subsequent life event. HorizonBench evaluates whether language models can perform this task. Each benchmark item presents a 5-option multiple-choice question embedded within a 6-month conversation history averaging ~4,300 turns and ~163K tokens. Pre-evolution preference values serve as hard-negative distractors, enabling diagnosis of belief-update failure.

Across 25 frontier models, the best achieves 52.8% and most score at or below the 20% chance baseline. When models err on evolved preferences, they select the pre-evolution distractor at rates significantly above chance, retrieving the user's originally stated preference without integrating the life event that changed it.

- **Paper**: [HorizonBench: Long-Horizon Personalization with Evolving Preferences](https://arxiv.org/abs/XXXX.XXXXX)
- **Dataset**: [stellalisy/HorizonBench on HuggingFace](https://huggingface.co/datasets/stellalisy/HorizonBench) (4,245 items, 360 users, 3 configs)

---

## Quick Start: Evaluate Any Model

### 1. Install

```bash
git clone https://github.com/stellalisy/HorizonBench.git
cd HorizonBench

# Using uv (recommended):
uv pip install .

# Or with pip:
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export OPENAI_API_KEY="your-key"          # for OpenAI models
export ANTHROPIC_API_KEY="your-key"       # for Anthropic models
export GEMINI_API_KEY="your-key"          # for Google models
```

### 3. Run evaluation

```bash
# Quick sanity check (10 items, ~30 seconds)
python evaluate.py --model gpt-4o --config sample

# Full benchmark (4,245 items)
python evaluate.py --model gpt-4o

# Any litellm-supported model works
python evaluate.py --model claude-sonnet-4-20250514
python evaluate.py --model gemini/gemini-2.0-flash

# Resume an interrupted run
python evaluate.py --model gpt-4o --resume

# Evaluate first 100 items only
python evaluate.py --model gpt-4o --max-items 100
```

### Example output

```
============================================================
  HorizonBench Results: gpt-4o
============================================================
  Overall:    42.3%  [40.8, 43.8]  (n=4245)
  Evolved:    28.1%  [26.1, 30.2]  (n=2135)
  Static:     56.4%  [54.3, 58.5]  (n=2110)
  Evo-Static gap: -28.3 pp
  Chance baseline: 20.0%

  By generator:
    sonnet-4.5            42.8%  [40.2, 45.3]  (n=1416)
    o3                    41.5%  [38.9, 44.1]  (n=1413)
    gemini-3-flash        42.7%  [40.1, 45.4]  (n=1416)
============================================================
```

Results are also saved to `results/results_{model}.jsonl` (one JSON object per item).

### Local and self-hosted models

For models running on vLLM, ollama, TGI, or any OpenAI-compatible endpoint, set the base URL and use litellm's provider prefix:

```bash
# Ollama
export OLLAMA_API_BASE="http://localhost:11434"
python evaluate.py --model ollama/llama3

# vLLM or any OpenAI-compatible server
export OPENAI_API_BASE="http://localhost:8000/v1"
python evaluate.py --model openai/my-local-model

# Together AI, Groq, etc.
export TOGETHER_API_KEY="your-key"
python evaluate.py --model together_ai/meta-llama/Llama-3-70b-chat-hf
```

See [litellm providers](https://docs.litellm.ai/docs/providers) for the full list of supported backends.

### 4. Evaluate a custom model or method

For RAG systems, fine-tuned models, or any custom inference pipeline, use the Python API:

```python
from evaluate import load_benchmark, build_prompt, extract_letter, print_results

ds = load_benchmark(config="benchmark")  # or "sample" for 10 items

results = []
for item in ds:
    prompt = build_prompt(item)
    response_text = your_model(prompt)  # replace with your inference function
    predicted = extract_letter(response_text)
    results.append({
        "id": item["id"],
        "generator": item["generator"],
        "has_evolved": item["has_evolved"],
        "correct_letter": item["correct_letter"],
        "predicted_letter": predicted,
        "correct": predicted == item["correct_letter"],
    })

print_results(results, "your-model-name")
```

### 5. RAG and retrieval-based methods

Each benchmark item contains a ~163K token conversation history (30+ sessions over 6 months). For methods that chunk, embed, or selectively retrieve from this history rather than passing it in full, use `parse_conversations()` and `build_question()`:

```python
from evaluate import load_benchmark, parse_conversations, build_question, extract_letter

ds = load_benchmark(config="sample")
item = ds[0]

# Parse conversation into structured segments
segments = parse_conversations(item["conversation"])
# segments[i] = {"date": "2025-10-01T...", "scenario": "...", "turns": [{"role": "user", "content": "..."}, ...]}

# Build your retrieval index from segments
for seg in segments:
    for turn in seg["turns"]:
        your_index.add(turn["content"], metadata={"date": seg["date"], "role": turn["role"]})

# Retrieve relevant context for the question
question = build_question(item)
retrieved_context = your_retriever.query(question, top_k=20)

# Combine and call your model
prompt = retrieved_context + "\n" + question
response_text = your_model(prompt)
predicted = extract_letter(response_text)
```

The `build_prompt()` function concatenates the full history + question (the default long-context protocol). Use `build_question()` to get only the MCQ portion when you handle context separately.

---

## Repository Structure

```
HorizonBench/
├── evaluate.py                  # Evaluate any model on HorizonBench (CLI + Python API)
├── pyproject.toml               # Dependencies (uv pip install . / .[generate] / .[analysis])
├── requirements.txt             # Minimal eval-only deps for pip users
│
├── src/                         # Data generation pipeline
│   ├── main.py                  # Entry point: python src/main.py --model gpt-4o
│   ├── timeline_generator.py    # Core timeline orchestrator
│   ├── benchmark_generation.py  # Counterfactual benchmark item generation
│   ├── benchmark_prompts.py     # Prompt assembly from generated timelines
│   ├── revalidate_history.py    # 5-LLM consensus history filter
│   ├── pipeline/                # End-to-end generation pipeline
│   ├── causal_framework/        # User, event, preference, intent models + evolution
│   ├── conversation/            # Conversation generation with preference annotations
│   ├── libraries/               # User, event, preference content libraries
│   ├── llm/                     # LLM client (git submodule → github.com/stellalisy/llm)
│   └── config/                  # Default configuration
│
├── scripts/                     # Analysis and reproduction scripts
│   ├── plot_model_accuracy.py   # Generate paper figures
│   ├── analyze_controlled_v2.py # Controlled experiment analysis (Table 2)
│   ├── bootstrap_ci.py          # Bootstrap confidence intervals
│   ├── stat_tests_controlled.py # Statistical significance tests
│   ├── analyze_accuracy.py      # Per-model accuracy table
│   ├── compute_annotator_agreement.py  # Human evaluation agreement
│   └── build_hf_dataset.py      # Build HuggingFace dataset from raw output
│
└── data/PAPI/                   # IPIP-NEO personality inventory (public domain)
```

---

## LLM Client Compatibility

### For evaluation (evaluate.py)

The evaluation script uses [litellm](https://docs.litellm.ai/) by default, which supports 100+ LLM providers through a unified interface. Set the appropriate environment variable for your provider and pass the model name.

### For custom inference (Python API)

If your method does not use litellm (e.g., a local model, a RAG pipeline, or a custom API), bypass it entirely by using the programmatic API shown above. Your inference function just needs to accept a string prompt and return a string response.

### Using src/llm directly

The `src/llm` submodule provides native clients for OpenAI, Anthropic, Google, and HuggingFace models. To use it with `evaluate.py`:

```bash
python evaluate.py --model gpt-4o --backend llm
```

For the data generation pipeline, `src/llm` is required (see setup instructions in the generation section below).

---

## Data Generation Pipeline

The generator constructs conversations from a structured mental state graph where life events drive preference changes through typed dependency edges. This inverts the standard approach of inferring mental state from conversations, providing ground-truth provenance for every preference change.

### Pipeline stages

1. **User instantiation**: sample user profiles with personality traits, demographics, social graphs
2. **Event sampling**: sample life events conditional on user state and event history
3. **Preference evolution**: causally evolve preferences based on life events
4. **Conversation generation**: produce outline then full conversation, with preference annotations
5. **Benchmark construction**: generate counterfactual response options and 5-LLM history validation

### Setup for generation

The generation pipeline requires additional dependencies and the `src/llm` submodule:

```bash
# Install generation dependencies
uv pip install ".[generate]"

# Initialize the llm submodule (skip if you cloned with --recursive)
git submodule update --init --recursive

# Set up API key
export OPENAI_API_KEY="your-key"
```

### Running the generator

```bash
# Generate a single event (fast test, ~30 seconds)
python src/main.py --model gpt-4o --single-event

# Generate one full user timeline (takes ~10-30 minutes depending on model)
python src/main.py --model gpt-4o

# Generate for multiple users
python src/main.py --model gpt-4o --users 10 --output-dir output/my_run
```

Key CLI arguments:

| Argument | Description |
|----------|-------------|
| `--model MODEL` | LLM model for generation (e.g., gpt-4o, claude-sonnet-4-20250514) |
| `--config PATH` | Path to config YAML (default: src/config/config.yaml) |
| `--users N` | Number of user timelines to generate |
| `--days N` | Timeline length in days (default: 180) |
| `--output-dir PATH` | Output directory (default: ./output) |
| `--single-event` | Generate one event + conversation instead of a full timeline |
| `--expression-type` | How preferences are expressed: explicit, implicit, or natural (default) |
| `--verbose` | Enable verbose logging |
| `--use-cache` | Resume from a previous run using cached libraries |

### Output format

The generator writes to the output directory:

```
output/
├── all_timelines.json              # all user timelines in one file
├── metadata.json                   # generation config and statistics
├── timelines/                      # one JSON file per user
│   └── timeline_FirstName_LastName_TIMESTAMP.json
├── benchmark/                      # counterfactual MCQ items (if enabled)
└── logs/
```

Each timeline JSON contains:

- **user**: generated user profile (demographics, personality traits, social graph)
- **event_record**: list of life events with dates, categories, and descriptions
- **preference_record**: preference state at each time step (with change provenance)
- **conversation_record**: list of conversations, each with date, event context, and dialogue turns

### Customizing for your own domain

The default config generates general-purpose chatbot conversations. To adapt the generator for a specific research domain, copy and modify the config file:

```bash
cp src/config/config.yaml src/config/my_config.yaml
# Edit my_config.yaml, then run:
python src/main.py --model gpt-4o --config src/config/my_config.yaml
```

The three key sections to customize in the config are:

**1. `preference_domains`** controls what types of user preferences exist and evolve:

```yaml
# Default (general chatbot):
preference_domains: ["emotional_support_style", "communication_intimacy", ...]

# Healthcare example:
preference_domains: ["treatment_approach", "medication_attitude",
                     "provider_communication_style", "wellness_priorities",
                     "diet_preferences", "exercise_routine", "mental_health_coping"]

# Education example:
preference_domains: ["learning_pace", "explanation_depth", "feedback_style",
                     "collaboration_preference", "assessment_format"]
```

**2. `category_weights`** controls what kinds of life events occur (weights are relative):

```yaml
# Default (general chatbot):
category_weights: {"emotional_social_support": 0.26, "romantic_interaction": 0.22, ...}

# Healthcare example:
category_weights: {"routine_checkup": 0.3, "symptom_onset": 0.2,
                   "lifestyle_change": 0.25, "medication_adjustment": 0.15,
                   "specialist_referral": 0.1}
```

**3. `assistant_config.example_roles`** controls the AI assistant personas:

```yaml
# Default:
example_roles: ["therapist", "tutor", "friend", "expert", ...]

# Healthcare example:
example_roles: ["doctor", "nurse", "nutritionist", "therapist", "pharmacist"]
```

Set `new_preference_domain_probability` and `new_event_category_probability` to 0 if you want to restrict generation strictly to your listed domains and categories. Otherwise the LLM may generate additional ones beyond your list.

The config file (`src/config/config.yaml`) contains inline documentation for every field.

---

## HuggingFace Dataset

The released dataset has three configs:

| Config | Rows | Description |
|--------|------|-------------|
| `benchmark` (default) | 4,245 | Full evaluation benchmark (5-option MCQ with conversation history) |
| `sample` | 10 | Curated subset for quick exploration and testing |
| `mental_state_graphs` | 360 | Structured user timelines with preference provenance |

### Loading the dataset

```python
from datasets import load_dataset

# Load benchmark items
ds = load_dataset("stellalisy/HorizonBench", "benchmark", split="test")

# Each item contains:
#   id, generator, user_id, conversation, correct_letter,
#   options, has_evolved, preference_domain, distractor_letter,
#   preference_evolution

# Load mental state graphs
graphs = load_dataset("stellalisy/HorizonBench", "mental_state_graphs", split="test")
```

### Linking benchmark items to mental state graphs

The `user_id` field (format: `{generator}/user_{N}`) links benchmark items to their source graph:

```python
graph_lookup = {g["user_id"]: g for g in graphs}

item = ds[0]
user_graph = graph_lookup[item["user_id"]]
# user_graph contains: events, preferences, preference_changes, conversations
```

---

## Analysis Scripts

The analysis scripts reproduce the paper's tables and figures. Install the analysis dependencies first:

```bash
uv pip install ".[analysis]"
```

These scripts reproduce the paper's figures and statistical tests from raw evaluation outputs. They expect the internal directory structure produced by the full evaluation pipeline (per-user directories with per-model JSONL results). They are included for transparency and reproducibility of paper results, not for use with `evaluate.py` output directly.

```bash
# Generate paper figures (accuracy bar chart, distractor rate, evolved vs. static)
python scripts/plot_model_accuracy.py --results-dir output/

# Controlled experiment analysis (Table 2)
python scripts/analyze_controlled_v2.py --results-dir output/

# Bootstrap confidence intervals
python scripts/bootstrap_ci.py --results-dir output/

# Statistical tests
python scripts/stat_tests_controlled.py --results-dir output/
```

---

## Citation

```bibtex
@inproceedings{li2025horizonbench,
  title={HorizonBench: Long-Horizon Personalization with Evolving Preferences},
  author={Li, Shuyue Stella and Paranjape, Bhargavi and Oktar, Kerem and Ma, Zhongyao and Zhou, Gelin and Guan, Lin and Zhang, Na and Park, Sem and Chen, Lin and Yang, Diyi and Tsvetkov, Yulia and Celikyilmaz, Asli},
  booktitle={Conference on Language Modeling (COLM)},
  year={2025}
}
```

## License

This project is released under the Apache 2.0 License.
