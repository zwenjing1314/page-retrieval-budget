# Visual RAG Toolkit：页面检索复现

本项目独立于旧 layoutpatch“证据区域排序”工作。
当前任务是固定模型与数据的页面检索流程复现，尚未开展新方法研究。

## 当前进度

P0 环境与模型/数据准备已完成，依据本轮人工执行反馈及保留记录登记：
独立 Conda 环境、GPU 版 PyTorch、Toolkit 依赖、Qdrant 服务、
Economics 数据和固定模型文件已完成准备及对应检查。

P1 正确性与工程诊断已通过，原始证据及历史失败记录保留。
当前进入 P2：先测量少量新增页面的资源开销，再完成全量编码缓存。
尚无正式 A/B/C 检索指标；全库索引 ID 一致性和实际存储验收仍待完成。

P1 验收入口：见 `toolkit-repro/records` 下最新的
`p1-closeout-*` 目录中的 `P1-checklist.md`。

协作方式：所有电脑操作由用户执行，指导者提供说明、命令并根据反馈验收。
每批完成后等待反馈，不自动推进。

## 实验范围与权威计划

以[复现执行计划与验收标准](复现执行计划与验收标准.md)为准；
该文件暂时保留根目录原路径。

- 模型：ColPali-v1.3，冻结推理，不训练。
- 数据：ViDoRe v2 Economics 完整 452 页、232 个问题、3628 条 qrels。
- A：完整向量单阶段检索。
- B：作者 legacy Conv1d 窗口 3 池化预筛 top256，再以完整向量重排。
- C：仅保存同一种池化向量的独立索引，进行单阶段检索。
- 返回 top100，使用精确搜索；三组复用同一套编码缓存与输入规则。

A/B 是 Economics 子任务上的方法流程复现；
C 是存储压缩研究所需的工程扩展，不称为新方法或论文主表完整复现。
不换模型、不扩展数据集、不搜索参数，不以涨分或四倍加速为通过条件。

## 当前环境与源码

只使用一个专用 Conda 环境：

```powershell
conda activate page-retrieval-budget
```

| 组件 | 当前选定版本 |
|---|---|
| Python | 3.11.x |
| PyTorch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| colpali-engine | 0.3.10，提交 cae32d372ce8ce40363f50283d81760d1e301424 |
| transformers / peft | 4.51.3 / 0.15.2 |
| accelerate / datasets | 1.6.0 / 3.5.0 |
| Qdrant Server / Client | 1.14.1 / 1.14.2 |

这是本机逐步验证的候选组合，不是作者公布的完整冻结环境。
具体包清单和镜像身份见 P0 记录。

作者源码以普通文件直接纳入本项目 Git 仓库，不使用子模块。
原始上游身份见 [toolkit-upstream.txt](./toolkit-repro/toolkit-upstream.txt)；
后续适配通过外层仓库的 diff 和提交历史记录。作者 LICENSE 保留。

作者源码位于 [toolkit-repro/visual-rag-toolkit](toolkit-repro/visual-rag-toolkit)，
固定提交为 `262e01575bd8d9599a98466dce6d12b88f45ea16`，
以 editable 方式安装。

约束文件：
[核心依赖 v1](toolkit-repro/core-constraints-v1.txt)、
[含 Qdrant 客户端的 v2](toolkit-repro/core-constraints-v2.txt)。

## 模型与数据位置

以下路径均相对于项目根目录；目录名包含固定 revision。

| 内容 | 路径 |
|---|---|
| Adapter 与 processor | [ColPali-v1.3](toolkit-repro/models/colpali-v1.3-b5c6dd62125326e6f0b540c1f7b36e901cdc0a11) |
| 基座 | [ColPaliGemma base](toolkit-repro/models/colpaligemma-3b-pt-448-base-30ab955d073de4a91dc5a288e8c97226647e3e5a) |
| 数据 | [Economics](toolkit-repro/data/economics_reports_v2-76fe40166ba07b1bf50457f5c6057cacdd045f10) |

各下载目录中的 `download-manifest.json` 保存文件身份和校验记录。
数据包含 corpus、queries、qrels、docs 四个配置，均使用 test split。
真实 doc-id 保留；不能用 corpus-id 推造物理页码。

Adapter 原始配置中的基座 revision 为 null。
P1 加载必须显式绑定已下载的固定基座，不能只固定 adapter 或自动使用远端默认版本。
原始 snapshot 保持不修改。

## Qdrant 服务与适配

- 地址：http://127.0.0.1:16333
- 专用容器：`prb-qdrant-1141`
- 专用 Docker volume：`prb-qdrant-1141-data`
- 镜像：`qdrant/qdrant:v1.14.1`，实际 digest 见部署记录。
- 合成检查已覆盖连接、多向量 MaxSim、FP16 配置读回和 Prefetch 接口。

参数适配：计划中的 `full_scan_threshold=0` 被该版本拒绝，
P0 测试改为允许的最小值 10，并记录原因。
通过 `m=0`、`indexing_threshold=0` 和查询 `exact=True` 保持无 HNSW、精确检索目标。
正式 A/B/C collection 仍须逐个读回验收，不能沿用测试通过结论。

## P0 记录入口

- [依赖安装与环境快照](toolkit-repro/records/p0-dependencies-20261106-194720)
- [Qdrant 合成检查](toolkit-repro/records/p0-qdrant-20261106-210049-ad8d2e)
- [Qdrant 部署、镜像身份与环境快照](toolkit-repro/records/p0-qdrant-deployment-20261106-210150)
- [Economics 数据完整性检查](toolkit-repro/records/p0-data-20261106-211941-03b34e)
- 模型与数据各自目录内的 `download-manifest.json`

上述为阶段记录，后续发生依赖或代码变化时保存新记录，不覆盖旧记录。
P0 数据检查已覆盖数量、唯一 ID、标签映射、query 0 和全部图像解码；
尚不代表 Toolkit loader 或正式索引已保留全部数据。

## P2 下一步

1. 新增编码少量真实页面，记录时间、PyTorch 显存峰值和进程 RAM 采样。
2. 根据预试验估算成本，补齐全部 452 页、232 题的可恢复缓存。
3. 缓存完整性通过后，依次执行正式 A、B、C；每组验收后再开始下一组。
4. 正式配置保持 top100；B 预筛 top256；三组使用相同输入与编码缓存。

## 文档导航与整理边界

文档归档已完成，入口见[文档导航](./docs/README.md)。

- [研究核查与首选方案](./docs/research/复现项目核查与首选方案.md)
- [研究交接说明](./docs/handoff/研究交接说明.md)
- [复现窗口起始消息](./docs/handoff/新窗口首条消息.txt)
- [历史启动提示词](./docs/handoff/新窗口启动提示词.md)
- [历史 README](./docs/handoff/history/)
- [整理前原文备份与迁移记录](./docs/archive/)

研究文档保留各自编写时点的事实和判断，不作为当前进度记录。
历史提示词中的旧行动建议不覆盖当前执行计划和人工操作要求。
执行计划继续保留根目录原路径。

本次仅整理文档，没有移动模型、数据、作者仓库、Conda 环境、
Qdrant volume 或已有 records，没有重跑下载、检查或实验。
旧 layoutpatch 的代码、环境、模型、缓存与实验记录保持隔离。
