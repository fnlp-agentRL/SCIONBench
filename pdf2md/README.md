# PDF 转 Markdown

原 `mineru_pdf_to_md.sh` 确实用于 PDF 转 Markdown，但它调用启智平台的 VLM 服务和未随仓库提供的本地 bridge。这里保留脚本名称和“输入 PDF 或目录、输出 Markdown 目录”的用法，改用 MinerU 官方本地 CLI，因此不需要启智环境或 `INF_API_KEY`。

## 安装

使用 Python 3.11–3.14，建议在 Linux 服务器上创建独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r pdf2md/requirements.txt
```

MinerU 4.x 的命令是 `mineru-kit parse`。首次运行 `standard` tier 时可能下载模型。部署前可用 `mineru-kit parse --help` 检查本机安装。安装和 tier 说明见 [MinerU 官方 Quick Start](https://opendatalab.github.io/MinerU/quick_start/)。

## 配置和运行

仓库根目录 `.env` 可设置 `PDF2MD_MINERU_BIN` 为 `mineru-kit` 的绝对路径，以及 `PDF2MD_TIER`（默认 `standard`）和可选的 `PDF2MD_OCR_MODE`。输入和输出目录由命令参数提供，无内置服务器路径。

```bash
./pdf2md/mineru_pdf_to_md.sh /path/to/paper.pdf /path/to/output
./pdf2md/mineru_pdf_to_md.sh /path/to/pdf-directory /path/to/output
```

目录模式递归查找 PDF，并在输出目录保留相对目录结构。每个 PDF 对应一个 `.md` 文件；MinerU 生成的图片等附属文件也会保存在输出路径附近。扫描版 PDF 可在 `.env` 中设置 `PDF2MD_OCR_MODE=ocr`，纯文本层提取可使用 `PDF2MD_TIER=flash` 和 `PDF2MD_OCR_MODE=txt`。
