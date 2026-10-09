#!/bin/bash

# 如果某一步报错，脚本直接停止
set -e

# ==============================
# GPU 等待设置
# ==============================

# 只需要改这里：单位是 GB
GPU_MEM_GB=27

# 自动转换成 MiB，因为 nvidia-smi 输出的是 MiB
THRESHOLD=$((GPU_MEM_GB * 1024))

# 每隔多少秒检查一次 GPU
SLEEP_INTERVAL=60

# 监控哪张 GPU
GPU_ID=0

# ==============================
# 需要运行的命令
# 不需要的命令保持为空即可，例如 CMD3=""
# ==============================

CMD1="CUDA_VISIBLE_DEVICES=0 python /home/zch/project/gptq_zch2/optzch.py \
  /mnt/dfq/pretrained_model/opt-6.7b \
  wikitext2 \
  --wbits 4 \
  --groupsize 128 \
  --nsamples 128 \
  --act-order \
  --save-gptq /mnt/dfq/gptq_output/opt-6.7b-w4-g128-gptq_params.pt"

CMD2="CUDA_VISIBLE_DEVICES=0 python /home/zch/project/gptq_zch2/optzch.py \
  /mnt/dfq/pretrained_model/opt-6.7b \
  wikitext2 \
  --wbits 3 \
  --groupsize 128 \
  --nsamples 128 \
  --act-order \
  --save-gptq /mnt/dfq/gptq_output/opt-6.7b-w3-g128-gptq_params.pt"

CMD3=""

CMD4=""

CMD5=""

# ==============================
# 创建输出目录
# ==============================

mkdir -p /mnt/dfq/gptq_output

# ==============================
# 等待 GPU 空闲显存
# ==============================

echo "开始监控 GPU ${GPU_ID}..."
echo "条件：GPU ${GPU_ID} 空闲显存 >= ${GPU_MEM_GB} GB，即 ${THRESHOLD} MiB 时开始运行任务"
echo "检查间隔：${SLEEP_INTERVAL} 秒"

while true
do
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU_ID} | xargs)

    echo "$(date '+%Y-%m-%d %H:%M:%S') GPU ${GPU_ID} free memory: ${FREE_MEM} MiB"

    if [ "$FREE_MEM" -ge "$THRESHOLD" ]; then
        echo "GPU ${GPU_ID} 空闲显存满足条件，开始运行任务..."
        break
    fi

    echo "GPU ${GPU_ID} 当前空闲显存不足 ${GPU_MEM_GB} GB，等待 ${SLEEP_INTERVAL} 秒..."
    sleep ${SLEEP_INTERVAL}
done

# ==============================
# 运行命令函数
# ==============================

run_cmd_if_not_empty() {
    local CMD_NAME="$1"
    local CMD_CONTENT="$2"

    if [ -z "$CMD_CONTENT" ]; then
        echo "${CMD_NAME} 为空，跳过。"
    else
        echo "=============================="
        echo "开始运行 ${CMD_NAME}"
        echo "命令内容："
        echo "$CMD_CONTENT"
        echo "=============================="

        eval "$CMD_CONTENT"

        echo "${CMD_NAME} 运行完成。"
    fi
}

# ==============================
# 依次运行 CMD1 到 CMD5
# ==============================

run_cmd_if_not_empty "CMD1" "$CMD1"
run_cmd_if_not_empty "CMD2" "$CMD2"
run_cmd_if_not_empty "CMD3" "$CMD3"
run_cmd_if_not_empty "CMD4" "$CMD4"
run_cmd_if_not_empty "CMD5" "$CMD5"

echo "全部非空命令运行结束。"