# OmniColPress 定向核查；附 MAGIC 补充入口

核查日期：2026-11-05。用途：供《启动核查》及本轮“先复现现成项目”的选型引用。

本文件区分【官方论文报告】【官方仓库/模型卡静态核查】【资源推导】【未知】。未克隆仓库、安装依赖、下载权重或数据、训练、模型评分或运行该项目；未修改 layoutpatch。

## 1. OmniColPress：结论与需要纠正的推荐

**论文和作者代码真实存在，训练、建索引、检索评价入口已公开；而且作者已经发布训练好的视觉任务 checkpoint。不能说必须先训练才能复现推理。**

当前仍不建议将其作为 Windows、RTX 3080 Ti 约 12GB 显存上的第一个完整复现项目。理由是自定义模型及依赖链、2048 维索引和整库驻设备评分的资源风险，以及尚缺一套已核实的固定 ViDoRe 数据转换/评价配方。它适合作为已学习压缩方法的近邻和后续对照来源；这不是否定其研究价值。

### 1.1 已核实身份

| 项目 | 来源与身份 |
|---|---|
| 论文 | [Multi-Vector Index Compression in Any Modality](https://arxiv.org/html/2602.21202v1)，Hanxiang Qin 等，arXiv:2602.21202v1，2026-02-24 |
| 作者仓库 | [hanxiangqin/omni-col-press](https://github.com/hanxiangqin/omni-col-press) |
| 仓库固定版本 | `4a559677bbc8a3ea0c10322a721b52bb70d382ec`，由只读 `git ls-remote` 核实 |
| 代码许可 | 已读取 MIT LICENSE；copyright 2026 Hanxiang Qin |
| 官方模型集合 | [hltcoe / Multi-Vector Index Compression in Any Modality](https://huggingface.co/collections/hltcoe/multi-vector-index-compression-in-any-modality) |
| 视觉 AGC 权重 | [hltcoe/AGC_qwen2.5-vl_colpali](https://huggingface.co/hltcoe/AGC_qwen2.5-vl_colpali)，revision `14ba8fb11de7d15d5a87c7fa17e893bffcdd9020` |
| 视觉未压缩底座权重 | [hltcoe/ColBERT_qwen2.5-vl_colpali](https://huggingface.co/hltcoe/ColBERT_qwen2.5-vl_colpali)，revision `14a7bb3328187705ff153e3511a47f9abb144054` |

HF 元数据列出这两个模型的 safetensors 分片、config、processor/tokenizer 文件；本轮没有下载这些权重。集合还列出 `hltcoe/MemTok_qwen2.5-vl_colpali` 和 `hltcoe/SeqReSize_qwen2.5-vl_colpali`。所查入口未定位到可直接下载的页面 embedding 缓存，不能据此断言作者在任何位置都没有发布缓存。

### 1.2 方法与数据：论文报告，不是我们的复现

- H-Pool 是非参数层次池化，压缩操作本身不需要训练。
- SeqResize、MemTok 属于学习压缩；AGC 通过学习的通用 query token 获取注意力，选取代表位置，再按余弦相似度分组、按注意力聚合。
- 视觉底座基于 **Qwen2.5-VL-3B**，改为双向注意力；论文 §6.2 报告使用 `vidore/colpali_train_set` 训练 2 epochs，全局 batch 112、梯度累积 4、学习率 `1e-5`、BF16。
- 视觉实验使用 **ViDoRe v2**，表 3 对应 Biomedical、Economics、ESG Reports、ESG Human，并含多语言结果。本文没有自行补造子集规模或页面分组。
- 视觉输出维度是 **2048**；未压缩平均 1297 个向量，AGC 为 64 个。论文/模型卡报告 nDCG@5 平均：自身未压缩底座 60.0，H-Pool 56.4，AGC 56.7，即 AGC 保留自身底座约 **94.5%** 的该指标。不能把跨模态概述中的“97%”直接替换为视觉实验结果。
- 论文称完整视觉表示不能装入其 FastPlaid 实验配置，因此视觉比较统一用 brute-force flat multi-vector index。这不是已经证明所有视觉数据集都无法用 FastPlaid。
- README 的训练示例有 `deepspeed --num_gpus=4`，但没有据此确认论文实际 GPU 型号、数量和 GPU 小时；不能用示例命令冒充论文硬件记录。

### 1.3 真实代码入口及缺口

以下均读取了实际源码，不仅是目录名：

| 入口 | 已核实作用 | 复现前需要注意 |
|---|---|---|
| [src/train.py](https://github.com/hanxiangqin/omni-col-press/blob/4a559677bbc8a3ea0c10322a721b52bb70d382ec/src/train.py) | 训练入口 | 首轮使用已发布权重可以不重跑训练 |
| [src/build_index.py](https://github.com/hanxiangqin/omni-col-press/blob/4a559677bbc8a3ea0c10322a721b52bb70d382ec/src/build_index.py) | corpus 编码、汇集及建索引 | 先积累 CPU NumPy 列表，再 `pad_and_concat`；padding 与副本会增加内存 |
| [src/evaluate.py](https://github.com/hanxiangqin/omni-col-press/blob/4a559677bbc8a3ea0c10322a721b52bb70d382ec/src/evaluate.py) | query 编码、检索、指标 | 会排除无 relevant ground-truth 的查询，分母要核对 |
| [src/encoder/select_encoder.py](https://github.com/hanxiangqin/omni-col-press/blob/4a559677bbc8a3ea0c10322a721b52bb70d382ec/src/encoder/select_encoder.py) | AttentionSelectEncoder / AGC | 文档编码需要 attention；末层被设为 eager attention |
| [src/arguments.py](https://github.com/hanxiangqin/omni-col-press/blob/4a559677bbc8a3ea0c10322a721b52bb70d382ec/src/arguments.py) | 参数与默认值 | dtype 默认 float32，attention 默认 `flash_attention_2`，不能依赖默认值估计 12GB 可行性 |
| [src/utils/dist_utils.py](https://github.com/hanxiangqin/omni-col-press/blob/4a559677bbc8a3ea0c10322a721b52bb70d382ec/src/utils/dist_utils.py) | 单/多进程设备初始化 | `WORLD_SIZE > 1` 才启用分布式；不能说推理必须 4 卡 |

README 提供通用建索引和检索命令，但语料、图像目录和 qrels 使用 `path/to/...` 占位符。已核实的 `train_config.yaml` 示例路径返回 404；这只能证明该路径不能取得文件，不能证明仓库中不存在任何配置。

**尚未核实为开箱即用的内容：** 将指定 ViDoRe revision 的 corpus/queries/qrels 转为该项目输入格式的完整固定命令、论文各表的冻结配置、同指标口径验证及原始页面 embedding 缓存。

### 1.4 Windows 与 12GB 边界

【源码事实】

1. README 依赖含 DeepSpeed、FlashAttention、Triton、FAISS-GPU、FastPlaid 等，版本约束不构成完整 lock。
2. 自定义 Qwen 模型文件无条件导入 `liger_kernel.transformers.fused_linear_cross_entropy.LigerFusedLinearCrossEntropyLoss`，还导入 `einops`；所查 README 安装段未把 Liger 写入依赖。
3. `components_factory.py` 在模块级导入多个 index 类；`base_index.py` 导入 `faiss`，`fastplaid_index.py` 导入 `sqlitedict` 与 `fast_plaid.search`。**选择 `multivec` 不等于自动绕开这些导入依赖。**
4. factory 还会导入 RetrievalTrainer。推理入口没有完全隔离训练/其他索引依赖。
5. `MultiVecIndex.add_vectors` 将整个索引放到 `self.device`；torch 检索分块的是查询 batch，未按候选页分块整库。模型 batch=1 不能保证整库检索显存足够。
6. 建库/评价出现 `.cpu().detach().numpy()`，没有总是显式转 float32。BF16 到 NumPy 的兼容性需要实际验证，本轮没有把它断言为已发生的运行错误。
7. HF config 记录 BF16、hidden size 2048、36 层、16 query heads、2 KV heads、`transformers_version=4.57.3`。这是配置元数据，不是经过验证的环境锁。
8. 模型卡用仓库自定义 `AttentionSelectEncoder` 与 `Qwen2_5ForEmbedding` 加载，不能假设普通 `AutoModel` 调用就能忠实复现。

【资源推导】

- AGC 的 `64 × 2048 × 2 bytes = 256 KiB/页`，恰等于 ColPali 的 `1024 × 128 × 2 bytes` 图像 patch 裸向量。**“64 个向量”不等于对任意底座都更省存储。**
- 自身未压缩表示按平均 `1297 × 2048 × 2 bytes` 约为 **5.07 MiB/页**；若按这一平均规模处理 452 页，裸向量约 **2.24 GiB**，还未加 padding、模型、query、相似度中间张量及其他副本。452 页 AGC 裸向量约 **113 MiB**。这些是算术示例，不是相应子集的实测平均长度/峰值。
- 模型卡称 95.1% 压缩，分母是它自己的 1297×2048 底座；不能据此说其索引比旧 ColPali 缓存小 95.1%。
- 完整权重是否能在本机 batch=1 稳定编码、末层 eager attention 峰值、整库检索显存、CPU/RAM 峰值和耗时均未实测；不提供虚构 GPU 小时。

【评价口径】源码 Recall 对所有 relevant 页面计覆盖；所查 nDCG 使用线性 gain。若外部评测器使用指数 gain，分级标签时需要核对。某一数据集所有正例同一等级时，这两种 gain 可能在归一化后抵消，不能泛称必然产生指标差异。

### 1.5 采用/暂缓判断

**采用为近邻和已发布权重的备选；暂缓作为首轮复现项目。** 若后续决定用它，先用原 checkpoint 完成单页、少量 query 的推理与评分正确性，再评估整库内存。若需要为 Windows 大幅改写依赖或索引器，应停止把它当成“低成本原样复现”，重新评估是否值得投入。当前没有启动这些工作。

## 2. 新找到的直接压缩项目：MAGIC

这是本轮额外检索得到的作者项目，**不是批准为第二条实验路线**。

### 2.1 身份与适用性

- 论文：[MAGIC: Marginal-Guided Compression with Optimal Transport for Efficient Visual Document Retrieval](https://arxiv.org/html/2609.21018v1)，Xu Yuan、Hua Liu、Wenqi Fan、Qing Li，arXiv:2609.21018v1，2026-09-17。论文正文给出作者仓库地址。
- 作者代码：[xandery-geek/MAGIC](https://github.com/xandery-geek/MAGIC)，核实 commit **`be20007d797e5049cd20a9fca870ff119e1413a8`**。
- 已读 LICENSE 是 **Apache 2.0**。README 顶端 CC BY 图标与底部代码许可措辞不同；引用代码许可以 LICENSE 为据，不混同论文许可。
- 直接目标是冻结多向量的离线压缩，最终用压缩向量 MaxSim 检索；无需新增训练网络，也不要求用完整向量在线重排。相较 Toolkit 的主两阶段流程，它更直接对应“压缩索引本身”。
- 简单例子：不是把每个 patch 等量平均，而是先用一批来自训练集的查询 token 估计哪些 patch 更常被匹配，再在给定代表数量下聚合。论文使用最优传输约束代表的覆盖；这不是已经证明本项目可压缩性预算或跨页共享有创新。

### 2.2 实际入口、数据与模型要求

| 入口 | 静态核实内容 |
|---|---|
| [scripts/evaluate.py 的 README 用法](https://github.com/xandery-geek/MAGIC/blob/be20007d797e5049cd20a9fca870ff119e1413a8/README.md) | 官方说明先缓存 frozen 模型的 query/document embeddings，使用 `--save-embeddings` |
| [scripts/magic/build_query_prior.py](https://github.com/xandery-geek/MAGIC/blob/be20007d797e5049cd20a9fca870ff119e1413a8/scripts/magic/build_query_prior.py) | 默认训练数据 `vidore/colpali_train_set`；约 10,000 条 query 的采样目标、20 页采样目标、200 个视觉字典代表、最终 1000 个 query-token；默认 query batch 64 |
| [scripts/magic/eval_magic.py](https://github.com/xandery-geek/MAGIC/blob/be20007d797e5049cd20a9fca870ff119e1413a8/scripts/magic/eval_magic.py) | 缓存位置、prior 文件、保留比例/固定数量、任务与语言、MTEB 指标参数公开 |
| [magic/magic_compressor.py](https://github.com/xandery-geek/MAGIC/blob/be20007d797e5049cd20a9fca870ff119e1413a8/magic/magic_compressor.py) | PyTorch 压缩实现；Farthest-first 初始化、query prior 权重、Sinkhorn 更新、hard/soft 输出 |
| [encoder/base.py](https://github.com/xandery-geek/MAGIC/blob/be20007d797e5049cd20a9fca870ff119e1413a8/encoder/base.py) | 默认 frozen `vidore/colqwen2.5-v0.2`；支持多个模型族；BF16；FlashAttention 可用则采用，否则不强制安装该 attention 实现 |
| [evaluation/mteb_adapter.py](https://github.com/xandery-geek/MAGIC/blob/be20007d797e5049cd20a9fca870ff119e1413a8/evaluation/mteb_adapter.py) | MTEB 封装、缓存读取、向量数及相似度统计入口 |

【论文报告】主实验是 ViDoRe v1（ColQwen2.5），ViDoRe v2 用于多语言迁移，也给出 ColPali、Jina-v4 等底座结果。默认 calibration 数据来源固定为训练集，论文明确其查询与评价查询隔离。不能用测试 qrels 或测试 queries 替代该 prior 来省事。

【源码事实】标准构建脚本先 `load_dataset(..., split="train")`，之后再采样；它没有在这条调用中设置 streaming。**最后只保存 1000 个 token，并不意味着只需读取或下载 1000 条轻量数据。** 源训练集还含图像，数据获取成本须另查。20 页和 10,000 query 是目标参数，代码按 source 分配并取整，实际数量还取决于可用 source/条数。

不需训练 MAGIC 专用权重，但需要 frozen 检索 checkpoint、对应全部页面/查询编码缓存和同一模型空间里的 calibration prior。README 给出生成 prior 的流程；在已查入口中未确认现成可下载的 prior 或 embedding 缓存。

依赖文件固定：Python 3.10 的 README 环境示例，torch 2.6.0、torchvision 0.21.0、transformers 5.3.0、colpali-engine 0.3.15、mteb 2.12.15 等。**固定版本清单不等于本机已经成功解析或运行**；本轮没有安装验证。

### 2.3 本机可行性的已知边界

- 相比 OmniColPress，常用压缩表示是 projected **128 维**，压缩核心是普通 PyTorch 操作；所读路径不要求引入 Qdrant。不能仅据这些事实保证 Windows/12GB 直接成功。
- 默认编码 batch 32，prior query batch 64，不能原样假设适配 12GB。下调 batch 是以后复现执行时的工程设置，但需记录与原配置差异、核对 batch 不改变 token 处理。
- **核心初始化仍有平方矩阵。** `farthest_first_indices` 先计算 `G = patches @ patches.T`。单页 N=1024、FP32 的 G 约 4 MiB；构建视觉字典时先将采样页 patch 拼在 CPU，再调用同一函数。若拼接后 20,000 个向量，G 约 1.49 GiB；只是示例，真实 N 由采样结果决定。
- 查询 prior 的每个 source 候选池被限制到 `per × 20`，代码明确以此控制平方 Gram 内存。因此不能把全部 10,000 query 的所有 token 都当成一次不受限平方矩阵来吓大成本；也不能忽略视觉字典那次跨采样页矩阵。
- 论文附录 D 将主优化工作量表示成 K×N，但目前作者代码的初始化额外构造 N×N，query prior 权重还会产生 L_cal×N；资源评估应按所选源码实际数组计。**不要把该压缩器直接推广为任意长文档拼接方案。**
- 论文附录 B.6 的延迟测量在一张 NVIDIA H20 上完成；穷举 MaxSim、batch=1、2000 页和 1000 条查询、每条重复五次。它报告的时间不是 3080 Ti 的时间；其“index size”按平均向量数×128×FP32 推算，为每千页十进制 MB，不等于实际索引文件字节。
- 当前代码保留数量用 `round(N * ratio)`，论文 §5.1 写 `ceil(rN)`；这是需要冻结记录的代码/论文差异。首次复现不能同时改预算取整、换底座和改 token 规则后宣称完全复现论文。

**判断：值得列入“直接压缩项目”的优先候选，适合在一个现成页面检索流程跑通后单独评估。** 它比“只在 ColPali 源码里借一个池化类”更接近完整压缩论文项目，但 calibration 数据与版本兼容问题仍在；不应为了增加一个名字立即并行下载其训练集或搭建第二套环境。

## 3. 给主报告的简短交接

1. 更正 OmniColPress：已公开训练好视觉 checkpoint，可跳过训练；不是只能多卡训练后才能评价。
2. OmniColPress 的 64×2048 与 ColPali 的 1024×128 在半精度下同为 256 KiB/页，跨底座比较必须算维度和实际字节。
3. OmniColPress 的单进程推理存在，但依赖导入与整库驻设备索引仍需解决，不建议首轮在 Windows 上押注。
4. MAGIC 是额外检索得到的真正作者压缩项目；有标准数据、编码缓存、prior 构建、压缩评价入口。它免训练但并非零 calibration 成本，也不是已在本机验证可运行。
5. 本文所有方法效果数字均为作者报告，所有资源数字均为明示维度下的算术推导；没有产生新实验结果。
