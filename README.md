# CoQuant: EGConv and EGComp

本仓库提供量化转换（EGConv）、码本融合（EGComp）、Fisher 统计、AWQ/GPTQ 参考生成与 WikiText-2 / C4 PPL 评测代码。模型、数据集、量化权重 `.pt` 不随仓库发布，由使用者在本地生成。

## 目录与使用入口

| 目录 | 用途 |
|---|---|
| `reproduction/` | 推荐入口：路径通过参数配置，从原始模型生成参考、Fisher、conversion、composition，再评测 |
| `SqueezeLLM/` | 原实验开发源码，包含 CPU 量化、融合、Fisher、分层导出及评测脚本 |
| `gptq/` | GPTQ 导出源码：`optzch.py`、`llamazch.py`、`llama_like_zch.py`，包含 `--save-gptq` |
| `awq_reference/` | 历史 SqueezeLLM 脚本依赖的 AWQ 源码及原许可证 |
| `gptq_reference/` | 历史静态分组参考生成所用的另一套 GPTQ 源码 |
| `requirements.txt` | 当前实验环境中的核心 Python 包版本 |
| `source_snapshot.json` | 整理前源码的 SHA256 清单；不是修改后的发布文件校验清单 |

`SqueezeLLM` 和两套 GPTQ 为研究源码存档，部分脚本仍包含历史机器路径、调度锁和实验专用配置；**不要直接批量运行其中的历史 shell 脚本**。新机器优先使用 `reproduction/scripts/`。原版 SqueezeLLM 的安装配置固定了旧 Transformers，请不要用其依赖覆盖本 README 的环境。

## 环境

参考环境：Linux、Python 3.10、PyTorch 2.9.1 + CUDA 12.8、Transformers 4.51.0。GPU 用于参考生成、Fisher 和 PPL；EGConv/EGComp 使用 CPU。参考机器具有 96 GB GPU 显存、188 GB RAM。13B 的 dense GPTQ `.pt` 约 26 GB，生成和合并需预留充足 RAM 与磁盘空间；`.pt` 是稠密参考导出，并不是紧凑的 3-bit/4-bit 部署格式。

```bash
git clone https://github.com/nyx681/coquant.git
cd coquant
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

模型由使用者按相应许可证下载为 Hugging Face 目录，需包含 config、tokenizer 和完整权重。推荐入口离线读取模型，不会自动下载受限模型。

支持的 `--model`：`llama1_7b`、`llama2_7b`、`opt67b`、`qwen3_8b_base`、`mistral7b`、`llama1_13b`、`llama2_13b`、`opt13b`。后三个为 40 层配置。Qwen 对应 Qwen3-8B-Base，Mistral 对应 Mistral-7B-v0.1；不能用 instruction-tuned 版本替换 base 模型。

## 数据准备

在数据根目录放置以下 **Hugging Face `Dataset.save_to_disk` 格式**的数据，每个数据集包含 `text` 列：

```text
data/
  wikitext2_test/       # WikiText-2 raw test，原始文档顺序
  c4_validation_shard/ # 实验使用的 C4 英文 validation shard，不是随意采样的子集
  c4_train/            # Fisher 校准来源
  pile_validation/     # AWQ / GPTQ 校准来源
```

可以把已获授权的数据文件加载后保存，例如：

```python
from datasets import load_dataset
# 本例只展示文件格式转换；输入文件和排列顺序须与目标实验一致。
ds = load_dataset('json', data_files='/absolute/path/c4-validation.json.gz', split='train')
ds.save_to_disk('/absolute/path/data/c4_validation_shard')
```

来源名称：WikiText-2 raw (`wikitext-2-raw-v1`)、C4 English、Pile validation (`mit-han-lab/pile-val-backup`)。下载后保留数据版本、shard 文件名、顺序和文件校验值。任意 C4 子集不等价于论文的评测集。

C4 token SHA256 已保存在 `reproduction/src/egquant/common.py` 中，准备与评测时都会严格校验。校验不符时不要删除校验后声称精确复现。模型/tokenizer 修订版本、数据顺序、tokenizer 特殊符号均会改变 token 流。尚未随此源码发布历史数据 shard 的完整下载清单；因此本仓库提供可执行流程，但不宣称仅凭任意同名数据集即可精确恢复全部历史数值。

## 完整流程（从原始模型开始）

以下命令使用 Bash；请替换三个绝对路径。所有步骤须使用相同的参数。`--bits` 可设为 3 或 4；更换模型或校准 seed 时使用新的工作目录，避免复用不匹配的缓存。

```bash
cd reproduction
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
args=(--model llama2_7b --bits 3 \
      --model-dir /absolute/path/llama-2-7b \
      --data-dir /absolute/path/data \
      --work-dir /absolute/path/work \
      --gptq-seed 42)

# 1. 原始矩阵分层导出
python scripts/prepare_sources.py "${args[@]}" --source original

# 2. 生成固定 token 缓存及校验信息
for dataset in wiki c4 fisher awq gptq; do
  python scripts/prepare_tokens.py "${args[@]}" --dataset "$dataset"
done

# 3. GPU 生成 AWQ / GPTQ 分层参考
python scripts/generate_baselines.py "${args[@]}" --method awq
python scripts/generate_baselines.py "${args[@]}" --method gptq

# 可选：导出单个 GPTQ 参数 .pt；后续 conversion 直接用分层参考，不需要这个合并文件
python scripts/export_gptq_pt.py "${args[@]}" \
  --output /absolute/path/llama2-7b-w3-g128-gptq_params.pt

# 4. GPU 收集 Fisher，CPU 保存统计
python scripts/collect_fisher.py "${args[@]}" --device cuda:0

# 5. CPU conversion；A/G 相互独立，有足够内存时可以并行
python scripts/run_quantization.py "${args[@]}" --method EGConv-A --workers 4
python scripts/run_quantization.py "${args[@]}" --method EGConv-G --workers 4
# AG 必须在两个 conversion 完整结束后运行
python scripts/run_quantization.py "${args[@]}" --method EGComp-AG --workers 4

# 6. GPU 测试；所有转换/融合均以 FP16 centers/scales 重建
for method in AWQ GPTQ EGConv-A EGConv-G EGComp-AG; do
  for dataset in wiki c4; do
    python scripts/evaluate_ppl.py "${args[@]}" --method "$method" \
      --dataset "$dataset" --device cuda:0 --batch-size 1
  done
done
```

量化支持 `--resume`，恢复时输入和配置必须匹配；参考生成不支持在半成品目录上直接重跑，请保留失败日志并使用新的工作目录。每次完成一个阶段都应检查 manifest/run 状态，不能把没有 PPL 输出的任务计为成功。

### 输出位置

```text
work/<model>/artifacts/original/
work/<model>/artifacts/fisher/
work/<model>/artifacts/w3/{awq,gptq}/
work/<model>/tokens/
work/<model>/outputs/w3/{EGConv-A,EGConv-G,EGComp-AG}/
```

`w4` 对应 4-bit。评测保存 `<dataset>_ppl.json`，包含 token 哈希、上下文长度、位宽和存储口径。PPL 使用长度 2048 的完整块、batch size 1，越小越好。末尾不完整块不参与计算。

## GPTQ `.pt` 格式和历史导出命令

推荐的 `generate_baselines.py` 保存 `layer_N.pt` 和 `params_N.pt`，后者包含完整 G128 scale、zero、maxq 及分组信息。`export_gptq_pt.py` 合并为以 Hugging Face 模块名为键的字典，每项包含重建 `weight` 和量化参数。

3-bit 的 `maxq=7`，4-bit 的 `maxq=15`。act-order 和 static groups 是不同设置：本文当前参考要求原始列顺序下连续 G128；每行每 128 个权重共享 scale、zero。不能把只保存每行一个 scale、丢失分组映射的旧导出当成完整 G128 参数。

保留的原始 OPT 导出入口也可使用：

```bash
# 从仓库根目录运行；PILEVAL_PATH 指向 datasets.load_dataset 可加载的数据源
export PILEVAL_PATH=/absolute/path/pile-validation-source
CUDA_VISIBLE_DEVICES=0 python gptq/optzch.py \
  /absolute/path/opt-13b wikitext2 --wbits 3 --groupsize 128 \
  --nsamples 128 --seed 0 --act-order --static-groups \
  --save-gptq /absolute/path/exports/opt-13b-w3-g128-gptq_params.pt
```

注意：此历史入口的 `wikitext2` 参数选择了 **Pile 校准 + WikiText-2 测试** 的实现，不是 WikiText-2 校准；导出之后会继续测试。普通 dense 参考生成无需编译 GPTQ CUDA 扩展，packed kernel/benchmark 属于另外的运行路径。

LLaMA 类模型还保留 `gptq/llamazch.py` 和 `gptq/llama_like_zch.py`，先检查各自 `--help`。跨架构推荐统一的 `reproduction` 入口。原静态参考生成代码见 `SqueezeLLM/quantization/prepare_extended_delta_sources_w3.py`；分层导出见 `chunk_gptq_params.py`，W3 连续分组重建验证见 `validate_w3_gptq.py`。

## 实验设置与复现边界

- Conversion：Fisher 加权，eta=0.5；G2048 共享 8（3-bit）或 16（4-bit）个中心，G128 一个 scale。
- Composition：合并两个 conversion 的候选中心，执行选择和细化；不是逐元素 MinError 直接拼接。
- Centers/scales：FP16；重建评测模型仍是浮点模型，名义 bit 不等于 `.pt` 文件大小。
- Portable 默认 AWQ/GPTQ 校准：128 × 2048，seed=42；Fisher：100 × 512，seed=0。
- **历史实验并非统一 seed**：最近 LLaMA-1-13B、OPT-13B W3 GPTQ 参考使用 seed=0，使用 `--gptq-seed 0` 记录这一设置。仅修改 seed 不足以保证与历史不同入口的采样、数值路径完全一致。
- 13B 配置已加入 portable 入口，但没有在发布时重跑全模型验证。不同来源的 legacy AWQ scale-search 和采样实现也不能默认视为完全等价。精确复现须对照模型、token、参考权重哈希。
- 原实验额外的 Fisher-only、RTN 及 AR/GR 融合源码保留在 `SqueezeLLM/quantization`；本推荐流程覆盖 A/G/AG 主实验，不声称为这些额外实验提供已验证的统一命令。

## 检查与第三方来源

```bash
cd reproduction
python -m unittest discover -s tests -v
python scripts/export_gptq_pt.py --help
```

这些是 CPU 合成回归测试，不替代全模型 GPU 复现实验。

SqueezeLLM、GPTQ 的原 LICENSE 均保留。AWQ/GPTQ/SqueezeLLM 派生代码的许可证文本还在 `reproduction/src/egquant/_third_party.py` 与 `licenses/`。模型、数据集需另行遵循原许可。本仓库没有重新授予模型权重或第三方数据的分发许可。
