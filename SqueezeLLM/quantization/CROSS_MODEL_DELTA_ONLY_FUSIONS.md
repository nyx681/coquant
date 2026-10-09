# LLaMA-2-7B / OPT-6.7B：去 scale、去外层 1+ 后的四项对照

对应用户 2026-09-20 要求，沿用已完成 LLaMA-1 对照的口径：

1. AWQ-delta 非均匀：S=N(d)，W4-G128。
2. GPTQ-delta 非均匀：S=N(d)，W4-G128。
3. 同组两个 16 中心码本的并集，按普通平方误差精确选 16 个。
4. 同一并集，按输入 activation 加权平方误差精确选 16 个。

当前两侧生产任务和全部检查点直接复用：
`/mnt/data/zch/quantization_experiments/llama2_opt_delta_only_20260920_095736/{llama2,opt}/remove_one/{awq_delta,gptq_delta}`。
不重新拟合单独方法，不使用原版含 scale 的中心，不使用 S=1+N(d) 的中心。

## 目标与 activation

每组原始 128 个标量权重为 x_j，两套候选中心并集为 U。
从 U 中选 16 个中心 C，最小化 sum_j s_j min_(c in C) (x_j-c)^2。
普通融合 s_j=1；activation 融合 s_j=mean_t |X_tj|。
使用已有 `group_center_selection.select_centers` 的同一精确一维动态规划。
没有后续 Lloyd 更新；不对中心本身进行聚类；不再引入 AWQ scale 或 delta 系数。

activation 来自各模型原始浮点 Linear 输入，使用同一 LLaMA-1 校准函数：
本地 Pile validation，shuffle seed=42，接受 512 条非空且长度不超过 512 token 的文本，
拼接后分成完整 512 token 块。各模型 tokenizer 不同，实际 token 总数以 calibration.json 为准。
统计 abs(input) 的通道均值，按块 FP32 求和、FP64 累加。校准不使用 WikiText-2 test。
保存 activation_mean_abs.pt、原文哈希/token 哈希以及 activation_summary.json，后续可以复用。

## 执行与检查点

`run_cross_model_four_ppl_suite.py` 启动两个模型的
`run_cross_model_delta_only_fusions.py`。各子进程共享原始权重读取，一次遍历生成两个独立目标的中心。
每个模块的两个源码本落盘后即可计算，不等待单独方法 PPL。
已有 endpoint 任务继续运行；默认每模型 4 个融合 CPU worker，endpoint suite 完成后每模型增至 16。
校准和完整 PPL 使用 GPU，融合 DP 使用 CPU。两模型新任务的 GPU 阶段通过文件锁串行；CPU 阶段并行。

每个模块保存 unit/centers 和 activation/centers 的 NPZ，含中心、损失和输入哈希。
逐组验证选中中心属于候选并集，且在相同目标下不劣于两个候选码本。
恢复时检查代码哈希、配置和每模块输入哈希；兼容检查点不重新计算。
完整模型只在校准、评估时加载，选择中心阶段按层读取原始权重，减少常驻内存。

WikiText-2 test PPL：既有 awq_compatible_ppl_eval，block_size=2048，batch_size=8，
FP16 展开码本，nearest-original labels 与 LLaMA-1 一样采用 float64 距离。
最终 comparison.md/json/csv 包含两个模型各四项 PPL；缺失结果用空值，不用局部误差替代 PPL。

## 验证

使用两模型真实 layer 0 的 q 和 down 模块小组，对比已有 LLaMA-1 两个融合脚本：
两种目标的中心逐值相同、代价浮点容差内相同；activation 全 1 时等同普通融合。
实际全模型循环另校验原始 chunks 与模型权重一致、所有 activation 通道/token 数、中心形状及有限性。
