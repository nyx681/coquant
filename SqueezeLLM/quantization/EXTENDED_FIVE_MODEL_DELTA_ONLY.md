# 五个新增模型：S=N(d) 单独量化与两种精确融合

模型：LLaMA-1-13B、LLaMA-2-13B、OPT-13B、Qwen3-8B-Base、Mistral-7B-v0.1。
每个模型四项：AWQ-delta、GPTQ-delta、系数 1 融合、系数 activation 融合。
统一 W4-G128，WikiText-2 test，2048 token 非重叠块，batch size 8。

单独方法使用 S=N(d)：无 AWQ-scale 重要性先验，无外层 1+；d 内的 0.1 下限仍保留。
沿用既有模块输入通道相对 RMS delta、逆误差映射、正值中位数归一化以及 16 类加权 K-means。
融合读取同组两侧非均匀中心，以原始 128 个权重计算候选并集中的精确 32 选 16。
activation 系数为原始浮点模型 Linear 输入的 mean(abs(x))；使用 Pile validation 校准，
不使用测试集优化中心，不做选后中心更新。

## 参考权重与精度

新模型尚无可复用的本轮参考 chunks，先用 `prepare_extended_delta_sources.py` 生成。
AWQ 参考调用 llm-awq_zch0722 的 `_fake_awq_candidate`，与该项目 `--mix_sources awq`
单来源分支一致：原始权重乘缓存的通道 scale，W4-G128 非对称量化后除回 scale。
使用项目现存对应模型 W4-G128 搜索缓存；GQA 中没有 scale 记录的模块按该分支使用单位 scale，并记录模块名。
这里的 scale 只参与 AWQ 黑盒参考的生成，不加入 delta K-means 的重要性。
这沿用项目 Qwen3/Mistral 参考导出脚本的来源；不宣称重建的参考与其他历史分支生成的 LLaMA-2 chunks 逐位一致。

GPTQ 使用本地 gptq_zch0722 核心：128 条 Pile validation 校准样本，各 2048 token，seed 0；
percdamp=.01，act_order=True，static_groups=True，非对称 W4-G128。
LLaMA-like 使用 qkv/o/up+gate/down 分阶段 GPTQ；OPT 同层所有 Linear 一起统计。
记录每组 scale/zero/g_idx，逐模块校验原始列顺序的 G128 重建。
逐层保存 GPTQ 权重与参数，恢复时复用完成层并重新传播校准激活，避免重做量化。

保持每个原始模型本身的权重精度：Qwen3-8B-Base 为 BF16，其余以实际 config/权重为准。
同一个模型四项保持同样精度；中心仍是 FP32，重建 labels 使用 float64 距离。
FP16 与 BF16 的 AWQ 单来源导出一致性、GPTQ 静态分组重建均已通过小型矩阵检查。
delta 映射与融合核函数和前一批实现一致。

## 任务调度与复用

`run_extended_four_ppl_suite.py` 持续读取 manifest.json。2026-09-21 调整为分阶段调度：
参考权重准备不占 CPU 模型流水线名额，可与现有中心计算并行；GPU 阶段等待共享锁与空闲显存。
最多同时运行两个 CPU 模型流水线，避免同时常驻十份模型及过量 CPU worker。
调度器支持 --resume 接管仍在运行的子进程，不重启既有量化任务。
每模型先准备参考，然后两个 endpoint 各 6 个 CPU worker，融合初始 4 个 worker；
本模型两个 endpoint 完成后融合增至 16 个 worker。
GPU 阶段使用共享文件锁；已有两个模型可先推进，缺模型路径的条目明确显示 waiting_for_model。
更新 manifest 的模型路径后，后台调度器会自动检测并安排。
原始权重、参考 chunks、中心、校准统计、PPL、日志均保存，不删除历史实验。
源脚本与输入配置有哈希/指纹校验。汇总 comparison.md/json/csv 共五模型二十行，未完成 PPL 留空。

三个 13B 路径现已更新为 /mnt/data/dfq/Llama-13b、Llama2-13b、opt-13b。
五模型任务均已接入；GPU 等待状态以各自 sources/run.json 为准，不将等待资源描述为正在计算。
