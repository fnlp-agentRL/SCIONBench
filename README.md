# ScionBench: PDF to research graph to evaluation

[中文版](README.zh-CN.md)

This repository turns a paper PDF into Markdown, extracts a structured research graph, prepares test inputs, runs a research agent with Codex, and judges its reports. The three stages are in `pdf2md/`, `extraction/`, and `pipeline/` (which also contains `pipeline/judge/`). The paper, its required research data, model access, and API credentials are supplied by the user; they are not included in this repository.

## 1. Install and configure

Use a Linux server with Bash and Python 3.11–3.14. This range satisfies MinerU 4.x; the pipeline itself needs Python 3.10 or newer. Install and authenticate Codex CLI separately. The full judging stage also needs an accessible OpenAI compatible Chat Completions service. Replace the example clone URL below with your repository URL.

```bash
git clone https://example.com/your/scibench-github.git scibench-github
cd scibench-github
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r pdf2md/requirements.txt -r pipeline/requirements.txt
cp .env.example .env
```

Edit `.env` for your server. Set `CODEX_BIN` and `PDF2MD_MINERU_BIN` to executable paths if `codex` or `mineru-kit` are not on `PATH`. Confirm Codex is authenticated with `codex login status` (or `<path-to-codex> login status` if it is not on `PATH`). MinerU's first model backed parse may download weights. See [MinerU installation](https://opendatalab.github.io/MinerU/quick_start/). `.env` is excluded by `.gitignore`; keep API keys out of commits.

The examples below use a workspace at `/work/scibench`. Replace this with paths on your server. Keep the same paper name under `md/` and `graphs/` so the pipeline can find the Markdown images referenced by the graph.

```text
/work/scibench/
├── pdfs/paper-001.pdf
├── md/paper-001/paper-001.md       # plus MinerU image assets
├── graphs/paper-001/
│   ├── paper_graph.json
│   ├── validation_report.json
│   └── data/                       # manually downloaded research data
└── results/                        # created by the pipeline
```

## 2. Convert PDF to Markdown

```bash
mkdir -p /work/scibench/md/paper-001
./pdf2md/mineru_pdf_to_md.sh \
  /work/scibench/pdfs/paper-001.pdf \
  /work/scibench/md/paper-001
```

The script also accepts a directory of PDFs and searches it recursively. It uses the local MinerU 4.x `mineru-kit parse` command. Configure `PDF2MD_TIER` in `.env` (default `standard`); scanned PDFs may need `PDF2MD_OCR_MODE=ocr`. Inspect the produced Markdown and images before extraction. See [pdf2md/README.md](pdf2md/README.md).

## 3. Extract and validate one graph

```bash
./extraction/run_one_paper.sh \
  /work/scibench/md/paper-001 \
  /work/scibench/graphs/paper-001
```

The first argument is a **paper directory**, not an individual `.md` file. It should contain the Markdown and all referenced images. The script uses the authenticated Codex CLI to create `paper_graph.json`, then runs the bundled validator and writes `validation_report.json`. Continue only when the command ends with `VALID`. `SCIONBENCH_MODEL` and `SCIONBENCH_MODEL_REASONING_EFFORT` in `.env` optionally override Codex defaults for this stage.

## 4. Download research data

Keep the paper's `images/` directory in the Markdown paper directory. The pipeline locates figures under `PIPELINE_SOURCE_MD_ROOT` and copies them into the graph and Markdown test artifacts during preprocessing.

**External data must be placed manually.** The graph extraction does not download datasets, code, weights, or other research assets. Read the paper and its graph to identify the required resources, obtain them from the appropriate source, and place them under a `data/` directory **beside `paper_graph.json`**:

```text
/work/scibench/graphs/paper-001/paper_graph.json
/work/scibench/graphs/paper-001/data/<downloaded files and directories>
```

Preserve the file names and directory structure expected by the experiment. The pipeline copies this `data/` tree into each Markdown test input. If `data/` is absent, the pipeline can still run its preprocessing, but data dependent tests will lack their inputs. Keep the source Markdown and its `images/` directory available for figure handling.

## 5. Run tests and judging

Fill these values in the root `.env` for the layout above (use your actual paths):

```dotenv
PIPELINE_MODE=single-paper
PIPELINE_INPUT_DIR=/work/scibench/graphs/paper-001
PIPELINE_PAPER_EXTRAC_ROOT=/work/scibench/graphs
PIPELINE_SOURCE_MD_ROOT=/work/scibench/md
PIPELINE_PREPROCESS_OUTPUT_DIR=/work/scibench/results/preprocess
PIPELINE_AGENT_OUTPUT_DIR=/work/scibench/results/agent
PIPELINE_JUDGE_OUTPUT_DIR=/work/scibench/results/judge
PIPELINE_MODEL=your-codex-model-id
PIPELINE_STOP_AFTER=8
JUDGE_MODEL=your-judge-model-id
JUDGE_API_ENDPOINT=https://example.com/v1/chat/completions
AIGW_API_KEY=your-judge-service-key
```

Run the complete pipeline with one command:

```bash
./pipeline/run.sh
```

The pipeline merges and splits the graph, prepares four research test modes, copies `data/` and figure assets, runs Markdown based Codex tests, then judges the reports. It uses temporary JSON during preprocessing and removes the split JSON afterwards; `evaluation.json` remains as held-out judge data. There is no JSON test or judge lane. In `single-paper` mode, output roots receive a subdirectory named after the paper. Use `PIPELINE_STOP_AFTER=4` to verify preprocessing without model calls; set it back to `8` for the full run. Set `PIPELINE_RESUME=1` to skip completed reports on a later run. See [pipeline/README.md](pipeline/README.md) for other modes and settings.

The full model and judge stages can consume substantial time and API usage. Check Codex authentication, model IDs, service access, data files, and output space before starting them.
