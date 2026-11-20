# ScionBench：从论文 PDF 到科研图与评测

[English](README.md)

本仓库提供完整流程：PDF 转 Markdown、单篇论文抽取为科研图、准备测试输入、通过 Codex 运行科研任务，以及评估报告。对应目录为 `pdf2md/`、`extraction/`、`pipeline/`；评估代码和提示词位于 `pipeline/judge/`。论文、实验所需数据、模型访问权限和 API 密钥均由使用者自行准备，仓库不包含这些资源。

## 1. 安装与配置

建议使用 Linux、Bash 和 Python 3.11–3.14；该版本范围适用于 MinerU 4.x，流水线本身要求 Python 3.10 或更新版本。另需安装并登录 Codex CLI。完整评估还需要可访问的 OpenAI compatible Chat Completions 服务。请将下方示例 clone 地址换成你的实际仓库地址。

```bash
git clone https://example.com/your/scibench-github.git scibench-github
cd scibench-github
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r pdf2md/requirements.txt -r pipeline/requirements.txt
cp .env.example .env
```

编辑 `.env`：若 `codex` 或 `mineru-kit` 不在 `PATH`，分别将 `CODEX_BIN`、`PDF2MD_MINERU_BIN` 设为可执行文件的绝对路径。用 `codex login status` 检查 Codex 登录状态；若不在 `PATH`，用 `<Codex 可执行文件路径> login status`。MinerU 首次进行模型解析时可能下载权重，安装说明见 [MinerU 官方文档](https://opendatalab.github.io/MinerU/quick_start/)。`.env` 已被 `.gitignore` 排除，请勿提交密钥。

下文以 `/work/scibench` 为示例；请替换为自己的服务器路径。`md/` 与 `graphs/` 下使用相同论文目录名，便于流水线定位图引用的 Markdown 图片。

```text
/work/scibench/
├── pdfs/paper-001.pdf
├── md/paper-001/paper-001.md       # 以及 MinerU 生成的图片资源
├── graphs/paper-001/
│   ├── paper_graph.json
│   ├── validation_report.json
│   └── data/                       # 手动下载的实验数据
└── results/                        # 流水线创建
```

## 2. 将 PDF 转为 Markdown

```bash
mkdir -p /work/scibench/md/paper-001
./pdf2md/mineru_pdf_to_md.sh \
  /work/scibench/pdfs/paper-001.pdf \
  /work/scibench/md/paper-001
```

脚本也可接收 PDF 目录并递归查找文件，使用本地 MinerU 4.x 的 `mineru-kit parse` 命令。`.env` 中的 `PDF2MD_TIER` 默认是 `standard`；扫描版 PDF 可能需要 `PDF2MD_OCR_MODE=ocr`。抽取前请检查 Markdown 和图片是否完整。详见 [pdf2md/README.md](pdf2md/README.md)。

## 3. 抽取并验证科研图

```bash
./extraction/run_one_paper.sh \
  /work/scibench/md/paper-001 \
  /work/scibench/graphs/paper-001
```

第一个参数是**论文目录**，不是单个 `.md` 文件；该目录应包含 Markdown 及其引用的全部图片。脚本调用已登录的 Codex CLI 生成 `paper_graph.json`，随后运行仓库内的验证器并写入 `validation_report.json`。命令最后输出 `VALID` 后再继续。可在 `.env` 中用 `SCIONBENCH_MODEL` 和 `SCIONBENCH_MODEL_REASONING_EFFORT` 覆盖本阶段的 Codex 默认配置。

## 4. 手动下载实验数据

论文 Markdown 目录中的 `images/` 应保留在原处。预处理时，流水线会从 `PIPELINE_SOURCE_MD_ROOT` 下定位图片，并复制到图和 Markdown 测试产物目录。

**外部数据需要手动准备。** 图抽取不会下载数据集、代码、模型权重或其他实验资源。请根据论文和抽取结果确定所需文件，自行从合适来源下载，并将它们放在与 `paper_graph.json` **同级的 `data/` 目录**：

```text
/work/scibench/graphs/paper-001/paper_graph.json
/work/scibench/graphs/paper-001/data/<下载的文件和子目录>
```

保留实验所需的文件名和目录结构。流水线会把该 `data/` 目录复制到各个 Markdown 测试输入目录。缺少 `data/` 时预处理仍可能完成，但依赖数据的测试没有所需输入。源 Markdown 目录及其 `images/` 也须保留，供图片整理步骤使用。

## 5. 运行测试与评估

按上述示例目录，在根目录 `.env` 填写以下配置；实际使用时换成自己的路径和模型：

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

配置好后，一条命令运行完整流水线：

```bash
./pipeline/run.sh
```

流水线会合并和拆分图、生成四种科研测试模式、复制 `data/` 和图片资源、使用 Codex 运行 Markdown 测试，最后评估报告。预处理期间会短暂使用 JSON，拆分 JSON 随后删除；留存的 `evaluation.json` 是评估端的隐藏答案。没有 JSON 测试或评估通道。`single-paper` 模式会在各输出根目录下自动增加论文目录名。可先设 `PIPELINE_STOP_AFTER=4` 检查预处理，再改回 `8` 运行全流程；后续运行可设 `PIPELINE_RESUME=1` 跳过已完成的报告。其他模式及配置见 [pipeline/README.md](pipeline/README.md)。

完整的模型测试和评估可能耗时并产生 API 用量。启动前请核对 Codex 登录、模型 ID、评估服务、数据文件及输出空间。
