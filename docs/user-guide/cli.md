# CLI Reference

The `lemat-synth` command-line tool lets you extract structured synthesis
procedures from materials science papers without writing any Python code.

```
lemat-synth extract <paper>   [key=value ...]
lemat-synth batch   <folder>  [key=value ...]
```

All settings — models, prompts, domain, output path — live in
`config/cli.yaml` at the repository root.  You can override any of them
directly on the command line using [Hydra](https://hydra.cc) `key=value`
syntax.

---

## Installation & setup

Install the package (one-time):

```bash
uv pip install -e .
```

Create a `.env` file in the repository root with the API keys you need:

```dotenv
GEMINI_API_KEY=...                  # Google Gemini models (synthesis default)
ANTHROPIC_API_KEY=...               # Claude models and with_performance=true
OPENAI_API_KEY=...                  # OpenAI GPT models
MISTRAL_API_KEY=...                 # Mistral models or pdf_extractor=mistral
# OpenRouter — one key slot per model family (use synthesis_api_key_env etc.)
OPENROUTER_QWEN_API_KEY=...         # qwen3.5-35b-a3b
OPENROUTER_KIMI_API_KEY=...         # moonshotai/kimi-k2.5
OPENROUTER_DEEPSEEK_API_KEY=...     # deepseek/deepseek-v3.2
```

The CLI loads `.env` automatically on every run.

---

## `lemat-synth extract` — single paper

```
lemat-synth extract INPUT_FILE [key=value ...]
```

`INPUT_FILE` can be a plain-text file (`.txt`), a Markdown file (`.md`
produced by a PDF extractor), or a PDF (`.pdf` — Docling is used
automatically).

### Examples

```bash
# Basic — uses all defaults from config/cli.yaml
lemat-synth extract paper.txt

# Write results to a custom folder
lemat-synth extract paper.txt output_dir=my_results/

# Use a different synthesis model
lemat-synth extract paper.txt synthesis_model=anthropic/claude-sonnet-4-6

# Domain-specific plot filtering (keeps only relevant figures)
lemat-synth extract paper.txt domain=catalysis

# Extract performance data from figures (requires ANTHROPIC_API_KEY)
lemat-synth extract paper.txt with_performance=true

# Use Mistral OCR for PDF extraction (requires MISTRAL_API_KEY)
lemat-synth extract paper.pdf pdf_extractor=mistral

# Route all models through OpenRouter
lemat-synth extract paper.txt \
    synthesis_model=openrouter/google/gemini-3-flash-preview \
    api_base=https://openrouter.ai/api/v1

# Override the synthesis extraction prompt
lemat-synth extract paper.txt \
    "prompts.synthesis_instructions=Extract only the primary synthesis route,
    ignoring any alternative procedures described."

# Use a PDF and override the judge model
lemat-synth extract paper.pdf \
    synthesis_model=gemini/gemini-2.5-pro \
    judge_model=anthropic/claude-sonnet-4-6
```

---

## `lemat-synth batch` — folder of papers

```
lemat-synth batch INPUT_DIR [key=value ...]
```

Processes every `.txt`, `.md`, and `.pdf` file in `INPUT_DIR` and writes
per-paper results to `output_dir/`.

### Examples

```bash
# Basic
lemat-synth batch papers/

# Custom output folder
lemat-synth batch papers/ output_dir=results/catalysis/

# Process at most 5 papers (useful for a quick test run)
lemat-synth batch papers/ max_papers=5

# Re-process everything, even papers already done
lemat-synth batch papers/ skip_existing=false

# Reduce parallelism if you hit rate limits
lemat-synth batch papers/ max_papers_parallel=2

# Use Mistral OCR for all PDFs in the folder
lemat-synth batch papers/ pdf_extractor=mistral

# Powerful model, catalysis domain, resume if interrupted
lemat-synth batch papers/ \
    synthesis_model=gemini/gemini-2.5-pro \
    domain=catalysis \
    skip_existing=true

# Different model per stage
lemat-synth batch papers/ \
    synthesis_model=openrouter/google/gemini-3-flash-preview \
    material_model=gemini/gemini-2.5-flash-lite \
    judge_model=anthropic/claude-sonnet-4-6 \
    api_base=https://openrouter.ai/api/v1
```

---

## Configuration reference

All keys below can be overridden from the command line.  Defaults are
read from `config/cli.yaml`.

### Models

| Key | Default | Description |
|-----|---------|-------------|
| `synthesis_model` | `gemini/gemini-2.0-flash` | Main extraction model — used for synthesis procedure extraction |
| `material_model` | `gemini/gemini-2.5-flash-lite` | Fast/cheap model for material-list extraction |
| `judge_model` | *(same as `synthesis_model`)* | Model for quality evaluation; defaults to the synthesis model via OmegaConf interpolation |
| `linker_model` | `gemini/gemini-3-pro-preview` | Model for linking plot series to materials (only used with `with_performance=true`) |
| `plot_model` | `claude-sonnet-4-20250514` | Claude model for extracting numerical data from plots (only used with `with_performance=true`) |
| `api_base` | `null` | Custom API base URL, e.g. `https://openrouter.ai/api/v1`. When set, applied to all DSPy models. |

Model strings follow the [LiteLLM](https://docs.litellm.ai/docs/providers)
convention: `{provider}/{model-name}`.  Common patterns:

```
gemini/gemini-2.0-flash
gemini/gemini-2.5-pro
anthropic/claude-sonnet-4-6
openai/gpt-4o
openai/gpt-4o-mini
openrouter/google/gemini-3-flash-preview  ← needs api_base=https://openrouter.ai/api/v1
openrouter/deepseek/deepseek-v3.2         ← needs api_base=https://openrouter.ai/api/v1
```

### API key selection (per component)

By default LiteLLM reads the API key from the standard environment variable
for each provider (`GEMINI_API_KEY` for `gemini/*` models, `ANTHROPIC_API_KEY`
for `anthropic/*`, etc.).  Use the `*_api_key_env` keys when you need to
override this — for example to route a model through a specific OpenRouter key
slot without changing the default for other components.

| Key | Default | Description |
|-----|---------|-------------|
| `synthesis_api_key_env` | `null` | Env var that holds the key for the synthesis model |
| `material_api_key_env` | `null` | Env var that holds the key for the material model |
| `judge_api_key_env` | *(same as `synthesis_api_key_env`)* | Env var for the judge model |
| `linker_api_key_env` | `null` | Env var for the linker model (`with_performance=true` only) |

**Allowed values** (the literal env-var name, not the key itself):
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`,
`OPENROUTER_QWEN_API_KEY`, `OPENROUTER_KIMI_API_KEY`, `OPENROUTER_DEEPSEEK_API_KEY`

```bash
# Use DeepSeek via OpenRouter for synthesis, keep material model on Gemini
lemat-synth extract paper.txt \
    synthesis_model=openrouter/deepseek/deepseek-v3.2 \
    synthesis_api_key_env=OPENROUTER_DEEPSEEK_API_KEY \
    api_base=https://openrouter.ai/api/v1

# Qwen for synthesis, Kimi for linking, each with its own OpenRouter slot
lemat-synth batch papers/ \
    synthesis_model=openrouter/qwen/qwen3.5-35b-a3b \
    synthesis_api_key_env=OPENROUTER_QWEN_API_KEY \
    linker_model=openrouter/moonshotai/kimi-k2.5 \
    linker_api_key_env=OPENROUTER_KIMI_API_KEY \
    api_base=https://openrouter.ai/api/v1
```

### Pipeline behaviour

| Key | Default | Description |
|-----|---------|-------------|
| `domain` | `generic` | Plot-relevance filter.  See the [Domain filtering](#domain-filtering) section below. |
| `with_performance` | `false` | When `true`, also extracts plot data and links it to synthesized materials.  Requires `ANTHROPIC_API_KEY`. |
| `output_dir` | `results` | Directory where per-paper result folders are written. |
| `pdf_extractor` | `docling` | PDF-to-text backend: `docling` (local, no API key) or `mistral` (Mistral OCR API, requires `MISTRAL_API_KEY`). |
| `figure_segmenter` | `dino` | Figure segmentation backend: `dino` (Grounding DINO + ResNet-152, 28 classes) or `florence` (Florence-2 + LoRA, binary quantitative/qualitative). |
| `florence_repo_id` | `amayuelas/plot-visualization-florence-2-lora-32` | HuggingFace LoRA adapter used when `figure_segmenter=florence`. |

### Batch-specific

| Key | Default | Description |
|-----|---------|-------------|
| `max_papers` | `null` | Stop after processing this many papers.  `null` = process all. |
| `skip_existing` | `true` | Skip papers that already have a result folder in `output_dir`. |
| `max_papers_parallel` | `4` | Number of papers processed concurrently. |

---

## Domain filtering

The `domain` setting controls which figures are kept when
`with_performance=true`.

| Value | Figures kept |
|-------|-------------|
| `generic` | All figures (no filtering) |
| `catalysis` | Conversion/selectivity vs temperature curves |
| `superconductors` | Resistivity ρ(T) and resistance R(T) plots |
| `electrochemistry` | Current/capacitance vs voltage curves |

---

## Customising prompts

Every prompt used during extraction can be overridden.  The full set of
prompt keys is in `config/cli.yaml` under the `prompts:` block.

```yaml
prompts:
  synthesis_system:          # System message injected before every synthesis call
  synthesis_instructions:    # Task instructions for the synthesis extractor
  paper_text_description:    # Description of the paper_text input field
  material_name_description: # Description of the material_name input field
  synthesis_output_description: # Description of the output field
  material_instructions:     # Task instructions for the material extractor
  material_input_description:   # Description of the material extractor input
  material_output_description:  # Description of the material extractor output
```

Override a prompt from the command line by quoting the value:

```bash
lemat-synth extract paper.txt \
    "prompts.synthesis_instructions=Focus on the sol-gel steps only. \
    Ignore characterisation and testing sections."
```

Or edit `config/cli.yaml` directly to make a permanent change.

---

## Environment variables

| Variable | Provider | When required |
|----------|----------|---------------|
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Google Gemini | Default synthesis and material models |
| `ANTHROPIC_API_KEY` | Anthropic Claude | `synthesis_model=anthropic/…` or `with_performance=true` |
| `OPENAI_API_KEY` | OpenAI GPT | `synthesis_model=openai/…` |
| `MISTRAL_API_KEY` | Mistral | `synthesis_model=openai/mistral-…` or `pdf_extractor=mistral` |
| `OPENROUTER_API_KEY` | OpenRouter | Any `openrouter/…` model string |

---

## LLM concurrency

Two independent knobs control how many API calls happen simultaneously:

| Setting | Default | How to change |
|---------|---------|---------------|
| Papers in parallel | 4 | `max_papers_parallel=2` on the command line |
| LLM calls per paper | env-driven | `LLM_SYNTHESIS_MAX_CONCURRENT_LLM_CALLS=8` in `.env` |

If you hit rate-limit errors, lower one or both values.

---

## Output structure

Results are written to `output_dir/<paper-name>/`.  Each folder contains
one JSON file per extracted material, plus optional performance files.

See the [Output Format](output-format.md) page for a full description of
the JSON schema.

---

## When to use the Hydra deployment scripts instead

The `lemat-synth` CLI covers the most common case — extracting from your
own papers.  For advanced workflows, use the scripts in
`examples/scripts/deployment/` directly:

- **Multi-LLM ensemble extraction** (`synthesis_extraction=multi_llm`)
- **Processing the full HuggingFace LeMat-Synth-Papers dataset**
- **Evaluation against human annotations** (`data_loader=annotation`)
- **Full Hydra sweep / multi-run** mode

Those scripts are configured with `examples/config/config.yaml` and
support the same Hydra override syntax.
