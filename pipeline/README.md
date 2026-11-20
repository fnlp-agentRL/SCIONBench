# ScionBench Markdown 测试流水线

本目录运行**单一 Markdown 通道**。原始 `paper_graph.json` 在预处理时短暂拆分为内部 JSON，用于生成 `agent_input.md` 和隐藏的评估文件；拆分 JSON 在预处理结束后删除，不会保留 `test-input/json`。Codex 只读取 `test-input/md/<mode>/<number>/agent_input.md`，评估器只处理 `md` 报告。`judge-input/<number>/<mode>/evaluation.json` 是隐藏答案，不属于 Agent 输入。

## 安装与配置

完整流程、PDF 和图抽取步骤见仓库根目录的 [英文 README](../README.md) 或 [中文 README](../README.zh-CN.md)。本流水线需要 Linux、Bash、Python 3.10+、已登录的 Codex CLI；评估图片处理还需 Pillow，见 `requirements.txt`。完整评估还需 OpenAI compatible 的评估模型服务。

从 `.env.example` 复制根目录 `.env`，填写：

- `PIPELINE_INPUT_DIR`：图目录，单篇模式下只应包含一个 `paper_graph.json`。
- `PIPELINE_PAPER_EXTRAC_ROOT`：图的共同根目录。
- `PIPELINE_SOURCE_MD_ROOT`：原论文 Markdown 和图片的根目录。
- `PIPELINE_PREPROCESS_OUTPUT_DIR`、`PIPELINE_AGENT_OUTPUT_DIR`、`PIPELINE_JUDGE_OUTPUT_DIR`：三个输出根目录。
- `CODEX_BIN`、`PIPELINE_MODEL`：Codex 可执行文件及配置中可用的模型 ID。
- `JUDGE_MODEL`、`JUDGE_API_ENDPOINT`、`AIGW_API_KEY`：运行评估阶段时填写。

保留 Markdown 目录中的 `images/`；把论文实验数据放在图文件同级的 `data/`：

```text
<paper-dir>/paper_graph.json
<paper-dir>/data/...
```

预处理会将 `data/` 复制到四个模式的 `test-input/md/<mode>/<number>/data/`。Markdown 目录与图目录最好使用相同的论文名，便于图片定位。

## 一键运行

```bash
./pipeline/run.sh
```

`PIPELINE_MODE=single-paper` 是默认值，也可设为 `batch` 或 `article`。`PIPELINE_STOP_AFTER=4` 只运行预处理；`5` 或 `7` 在 Markdown Codex 测试后停止；`6` 或 `8` 运行测试和评估。`PIPELINE_RESUME=1` 可跳过已有的非空报告。`PIPELINE_LAUNCHER=direct` 默认直接运行监督器，也可选择 `auto`、`tmux` 或 `systemd`。高级参数可追加在 `./pipeline/run.sh` 命令后。

预处理产物位于 `<preprocess-root>/<paper-name>/test-input/md/`，Codex 报告位于 `<agent-root>/<paper-name>/md/`，评估结果位于 `<judge-root>/<paper-name>/md/`。单篇模式会自动追加论文目录名。
