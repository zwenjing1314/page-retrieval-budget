# 练习副本说明（2080 Ti 云端练习）

## 来源（未修改的原始仓库）
- 项目：`kuangliu/pytorch-cifar`（经典 CIFAR-10 训练示例）
- commit：`49b7aa97b0c12fe0d4054e670403a16b6b834ddd`
- 原始副本保留在 `code/pytorch-cifar/`，未改动。
- 依赖：仅 torch + torchvision（README 标称 PyTorch 1.0+）。

## 运行副本相对原仓库的差异（只有 main.py 两处）
| 位置 | 原内容 | 练习副本 | 目的 |
|---|---|---|---|
| 参数区（第 22 行附近） | 无 `--epochs` | 新增 `--epochs`（默认 200，保持上游行为） | 练习时用 `--epochs 1` 缩短为 1 轮，控制时长与费用 |
| 训练循环（第 153 行） | `range(start_epoch, start_epoch+200)` | `range(start_epoch, start_epoch+args.epochs)` | 让上面的参数生效 |

两处均有 `PRACTICE PATCH` 行内注释，可 grep 定位。其余（`models/`、`utils.py`、`main.py` 其他部分）与原仓库逐字节一致。

## 为什么这个项目适合本次练习
- 原生支持 `--resume`：checkpoint 保存 `net/acc/epoch`，可练"保存 → 重载 → 断点恢复"全流程。
- 依赖极简：torch + torchvision，无 transformers/peft/额外大数据，图灵卡无 bf16 问题（脚本全程 fp32）。
- checkpoint 内容仅 tensor/int/float，PyTorch 2.6 的 `torch.load(weights_only=True)` 可正常加载。

## 明确边界
- 这是通用练习 demo，**不是**项目研究代码，不验证 ColPali / 压缩 / 检索，也不能代替 v1.3 验收。
- 不修改原研究项目，不上传整个项目，不下载 52.7GB 训练集。
- 练习目录在工作区，独立于项目 git 仓库；不涉及分支或 worktree。
