from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
import time
import subprocess
import threading
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq722
from awq.quantize.quantizer import (
    prepare_wqlinear0124_cuda_runtime,
    real_quantize_model_weight_0124,
    pseudo_quantize_model_weight722,
)
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from torch import nn
import tqdm
# from simple_logger import setup_simple_logging
from accelerate import Accelerator
from transformers import AutoTokenizer


_process_gpu_memory_tracker = None


def _run_nvidia_smi_query(args):
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip()


def _get_gpu_bus_id_to_index():
    output = _run_nvidia_smi_query(
        ["--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"]
    )
    if output is None:
        return {}
    mapping = {}
    for line in output.splitlines():
        idx_str, bus_id = [part.strip() for part in line.split(",", 1)]
        mapping[bus_id] = int(idx_str)
    return mapping


def _snapshot_process_gpu_memory_mb(pid):
    """Return {gpu_index: used_MiB} for the given PID only."""
    bus_to_idx = _get_gpu_bus_id_to_index()
    output = _run_nvidia_smi_query(
        [
            "--query-compute-apps=pid,gpu_bus_id,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return {}
    usage = {}
    for line in output.splitlines():
        pid_str, bus_id, used_mb_str = [part.strip() for part in line.split(",", 2)]
        if int(pid_str) != pid:
            continue
        gpu_idx = bus_to_idx.get(bus_id)
        if gpu_idx is None:
            continue
        used_mb = int(used_mb_str)
        usage[gpu_idx] = max(usage.get(gpu_idx, 0), used_mb)
    return usage


class _ProcessGpuMemoryTracker:
    """Poll nvidia-smi and track peak GPU memory for the current process only."""

    def __init__(self, pid, interval_sec=0.5):
        self.pid = pid
        self.interval_sec = interval_sec
        self._peak_mb = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._nvidia_smi_available = _run_nvidia_smi_query(
            ["--query-gpu=name", "--format=csv,noheader"]
        ) is not None

    def start(self):
        if not self._nvidia_smi_available:
            return
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _poll_loop(self):
        while not self._stop.is_set():
            for gpu_idx, used_mb in _snapshot_process_gpu_memory_mb(self.pid).items():
                with self._lock:
                    self._peak_mb[gpu_idx] = max(
                        self._peak_mb.get(gpu_idx, 0), used_mb
                    )
            self._stop.wait(self.interval_sec)

    def get_peak_mb(self):
        with self._lock:
            peaks = dict(self._peak_mb)
        snapshot = _snapshot_process_gpu_memory_mb(self.pid)
        for gpu_idx, used_mb in snapshot.items():
            peaks[gpu_idx] = max(peaks.get(gpu_idx, 0), used_mb)
        return peaks


def _iter_valid_cuda_devices():
    if not torch.cuda.is_available():
        return
    try:
        count = torch.cuda.device_count()
    except RuntimeError:
        return
    for i in range(count):
        try:
            with torch.cuda.device(i):
                torch.cuda.current_device()
            yield i
        except RuntimeError:
            continue


def start_process_gpu_memory_tracking():
    global _process_gpu_memory_tracker
    for i in _iter_valid_cuda_devices():
        try:
            with torch.cuda.device(i):
                torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            continue
    _process_gpu_memory_tracker = _ProcessGpuMemoryTracker(os.getpid())
    _process_gpu_memory_tracker.start()


def report_peak_gpu_memory():
    global _process_gpu_memory_tracker
    pid = os.getpid()
    print(f"\n========== Peak GPU Memory (PID {pid} only) ==========")

    if _process_gpu_memory_tracker is not None:
        _process_gpu_memory_tracker.stop()
        peak_mb = _process_gpu_memory_tracker.get_peak_mb()
    else:
        peak_mb = _snapshot_process_gpu_memory_mb(pid)

    if peak_mb:
        for gpu_idx in sorted(peak_mb):
            used_mb = peak_mb[gpu_idx]
            name = ""
            try:
                if torch.cuda.is_available() and gpu_idx < torch.cuda.device_count():
                    name = f" ({torch.cuda.get_device_name(gpu_idx)})"
            except RuntimeError:
                pass
            print(
                f"  GPU {gpu_idx}{name}: peak {used_mb / 1024:.3f} GiB "
                f"({used_mb:.1f} MiB)"
            )
    elif (
        _process_gpu_memory_tracker is not None
        and not _process_gpu_memory_tracker._nvidia_smi_available
    ):
        print("  [warn] nvidia-smi unavailable; cannot report per-process GPU memory.")
    else:
        print("  (no GPU memory attributed to this process)")

    valid_devices = list(_iter_valid_cuda_devices())
    if valid_devices:
        print("  PyTorch allocator peak (same process, CUDA tensors only):")
        for i in valid_devices:
            try:
                with torch.cuda.device(i):
                    torch.cuda.synchronize()
                    alloc = torch.cuda.max_memory_allocated()
            except RuntimeError:
                continue
            if alloc == 0 and i not in peak_mb:
                continue
            print(
                f"    GPU {i}: {alloc / (1024 ** 3):.3f} GiB "
                f"({alloc / (1024 ** 2):.1f} MiB)"
            )

    print("  (does not include memory used by other processes on the GPU)")
    print("====================================================\n")


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
parser.add_argument("--quant_type", type=str, default='awq', help="legacy single-source selector used when --mix_sources is not set")
parser.add_argument(
    "--mix_sources",
    type=str,
    default=None,
    help=(
        "comma-separated quantization sources to combine; fake supports "
        "awq,gptq,omni,w_spqr,rtn and real supports awq,gptq, awq,rtn, "
        "or gptq,rtn"
    ),
)
parser.add_argument("--gptq_params", type=str, default=None, help="path to GPTQ fake-quant params .pt")
parser.add_argument("--omni_params", type=str, default=None, help="path to OmniQuant fake-quant params .pt")
parser.add_argument("--spqr_params", type=str, default=None, help="path to SpQR exported params .pt or legacy --save directory")
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument("--dump_fake", type=str, default=None, help="save fake-quantized model")
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
parser.add_argument(
    "--vila-15",
    action="store_true",
    help="quantizing vila 1.5",
)
parser.add_argument(
    "--mseon",
    action="store_true",
    help="enable MSE evaluation (default: False, skip MSE calculation)",
)
parser.add_argument(
    "--save_model_path",
    type=str,
    default=None,
    help="save quantized model in HuggingFace format after PPL evaluation",
)
args = parser.parse_args()


def torch_load_with_progress(path, label=None, map_location="cpu"):
    start_time = time.time()
    file_size = os.path.getsize(path) / (1024 ** 3) if os.path.isfile(path) else None
    size_msg = f" ({file_size:.2f} GB)" if file_size is not None else ""
    label = label or os.path.basename(path)
    tqdm.tqdm.write(f"Loading {label} from {path}{size_msg} ...")
    with tqdm.tqdm(total=1, desc=f"loading {label}") as pbar:
        obj = torch.load(path, map_location=map_location)
        pbar.update(1)
    item_count = len(obj) if hasattr(obj, "__len__") else "unknown"
    tqdm.tqdm.write(f"Loaded {label}; items={item_count}; elapsed={time.time() - start_time:.1f}s")
    return obj

vila_10_quant_mode = ("llava" in args.model_path.lower() or "vila" in args.model_path.lower()) and not args.vila_15

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer

def layer_name_prefix(model):
    if model.__class__.__name__ == "OPTForCausalLM":
        return "model.decoder.layers"
    return "model.layers"


def format_layer_name(model, layer_idx, linear_name):
    return f"{layer_name_prefix(model)}.{layer_idx}.{linear_name}"


def layer_name_aliases(full_name):
    aliases = [full_name]
    if full_name.startswith("model.layers."):
        aliases.append("model.decoder.layers." + full_name[len("model.layers."):])
    elif full_name.startswith("model.decoder.layers."):
        aliases.append("model.layers." + full_name[len("model.decoder.layers."):])
    if full_name.startswith("model."):
        aliases.append(full_name[len("model."):])
    return list(dict.fromkeys(aliases))


def find_scale_for_layer(awq_results, full_name):
    """
    根据 layer 的 full_name 在 awq_results["scale"] 中查找对应的 scale
    
    参数:
        awq_results: awq_results 字典，必须包含 "scale" 键
        full_name: layer 的完整名称，例如 "model.layers.0.self_attn.q_proj"
    
    返回:
        tuple: (prev_op_name, scales, th_t) 如果找到，否则返回 None
        或者返回完整的元组 (prev_op_name, layer_names, scales, th_t)
    """
    if "scale" not in awq_results:
        return None
    
    for scale_item in awq_results["scale"]:
        if len(scale_item) >= 3:
            prev_op_name, layer_names, scales = scale_item[0], scale_item[1], scale_item[2]
            th_t = scale_item[3] if len(scale_item) > 3 else None
            
            # layer_names 可能是一个列表或单个字符串
            #print(f"During find_scale_for_layer, layer_names: {layer_names}")
            #print(f"During find_scale_for_layer, full_name: {full_name}")
            aliases = set(layer_name_aliases(full_name))
            if isinstance(layer_names, list):
                if any(name in aliases for name in layer_names):
                    return (prev_op_name, layer_names, scales, th_t)
            elif isinstance(layer_names, tuple):
                if any(name in aliases for name in layer_names):
                    return (prev_op_name, layer_names, scales, th_t)
            elif isinstance(layer_names, str):
                if layer_names in aliases:
                    return (prev_op_name, layer_names, scales, th_t)
    
    return None


def compare_awq_results(awq_results1, awq_results2, tolerance=1e-6):
    """
    比较两个 awq_results 是否一致
    
    参数:
        awq_results1: 第一个 awq_results 字典
        awq_results2: 第二个 awq_results 字典
        tolerance: 浮点数比较的容差（默认 1e-6）
    
    返回:
        dict: 包含比较结果的字典
    """
    differences = []
    identical_keys = []
    
    # 获取所有键
    all_keys = set(awq_results1.keys()) | set(awq_results2.keys())
    
    for key in all_keys:
        if key not in awq_results1:
            differences.append({
                'key': key,
                'issue': f'Missing in awq_results1',
                'type': 'missing_key'
            })
            continue
        
        if key not in awq_results2:
            differences.append({
                'key': key,
                'issue': f'Missing in awq_results2',
                'type': 'missing_key'
            })
            continue
        
        val1 = awq_results1[key]
        val2 = awq_results2[key]
        
        # 比较不同类型的值
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            # 比较张量
            if val1.shape != val2.shape:
                differences.append({
                    'key': key,
                    'issue': f'Shape mismatch: {val1.shape} vs {val2.shape}',
                    'type': 'shape_mismatch'
                })
                continue
            
            diff = torch.abs(val1 - val2)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            if max_diff > tolerance:
                differences.append({
                    'key': key,
                    'issue': f'Tensor values differ: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}',
                    'type': 'value_mismatch',
                    'max_diff': max_diff,
                    'mean_diff': mean_diff
                })
            else:
                identical_keys.append(key)
        
        elif isinstance(val1, list) and isinstance(val2, list):
            # 比较列表（通常是 scales_list）
            if len(val1) != len(val2):
                differences.append({
                    'key': key,
                    'issue': f'List length mismatch: {len(val1)} vs {len(val2)}',
                    'type': 'length_mismatch'
                })
                continue
            
            list_diffs = []
            for i, (item1, item2) in enumerate(zip(val1, val2)):
                if isinstance(item1, tuple) and isinstance(item2, tuple):
                    # scales_list 中的项通常是元组: (prev_op_name, layer_names, scales, th_t)
                    if len(item1) != len(item2):
                        list_diffs.append(f'Item {i}: tuple length mismatch')
                        continue
                    
                    # 比较元组中的每个元素
                    for j, (elem1, elem2) in enumerate(zip(item1, item2)):
                        if isinstance(elem1, torch.Tensor) and isinstance(elem2, torch.Tensor):
                            if elem1.shape != elem2.shape:
                                list_diffs.append(f'Item {i}, element {j}: shape mismatch')
                                break
                            diff = torch.abs(elem1 - elem2)
                            max_diff = diff.max().item()
                            if max_diff > tolerance:
                                list_diffs.append(f'Item {i}, element {j}: max_diff={max_diff:.2e}')
                                break
                        elif elem1 != elem2:
                            list_diffs.append(f'Item {i}, element {j}: value mismatch')
                            break
                elif item1 != item2:
                    list_diffs.append(f'Item {i}: value mismatch')
            
            if list_diffs:
                differences.append({
                    'key': key,
                    'issue': f'List values differ: {"; ".join(list_diffs[:3])}',
                    'type': 'list_mismatch',
                    'details': list_diffs
                })
            else:
                identical_keys.append(key)
        
        elif isinstance(val1, dict) and isinstance(val2, dict):
            # 递归比较字典
            sub_diffs = compare_awq_results(val1, val2, tolerance)
            if sub_diffs['different_keys'] > 0:
                differences.append({
                    'key': key,
                    'issue': f'Dict contains {sub_diffs["different_keys"]} different keys',
                    'type': 'dict_mismatch',
                    'sub_differences': sub_diffs['differences']
                })
            else:
                identical_keys.append(key)
        
        else:
            # 其他类型直接比较
            if val1 != val2:
                differences.append({
                    'key': key,
                    'issue': f'Value mismatch: {type(val1).__name__} vs {type(val2).__name__}',
                    'type': 'value_mismatch'
                })
            else:
                identical_keys.append(key)
    
    result = {
        'total_keys': len(all_keys),
        'identical_keys': len(identical_keys),
        'different_keys': len(differences),
        'differences': differences,
        'is_identical': len(differences) == 0
    }
    
    return result


def compare_model_states(model, saved_state_dict, model_state_dict=None):
    """
    比较当前模型状态与保存的状态，检查权重是否相同
    
    参数:
        model: 当前模型对象（恢复后的模型）
        saved_state_dict: 保存的模型状态字典（第一次 apply_awq722 后的状态）
        model_state_dict: 可选，保存的模型状态字典（用于验证恢复是否正确）
    
    返回:
        dict: 包含比较结果的字典
    """
    from awq.quantize.pre_quant import get_blocks, get_named_linears
    
    layers = get_blocks(model)
    
    differences = []
    identical_layers = []
    max_diff = 0.0
    max_diff_layer = None
    
    for i in range(len(layers)):
        named_linears = get_named_linears(layers[i])
        
        for n, m in named_linears.items():
            full_name = format_layer_name(model, i, n)
            
            if full_name in saved_state_dict:
                # 比较权重
                w_saved = saved_state_dict[full_name]  # 已经是 CPU tensor
                w_current = m.weight.data.cpu()
                
                # 计算差异
                diff = torch.abs(w_saved - w_current)
                max_diff_val = diff.max().item()
                mean_diff_val = diff.mean().item()
                
                if max_diff_val > 1e-6:  # 如果差异大于阈值
                    differences.append({
                        'layer': full_name,
                        'max_diff': max_diff_val,
                        'mean_diff': mean_diff_val,
                        'shape': tuple(w_saved.shape)
                    })
                    if max_diff_val > max_diff:
                        max_diff = max_diff_val
                        max_diff_layer = full_name
                else:
                    identical_layers.append(full_name)
                
                # 如果提供了 model_state_dict，也验证恢复是否正确
                if model_state_dict is not None and full_name in model_state_dict:
                    saved_weight = model_state_dict[full_name]['weight']
                    diff_with_saved = torch.abs(w_current - saved_weight)
                    max_diff_saved = diff_with_saved.max().item()
                    if max_diff_saved > 1e-6:
                        print(f"Warning: Layer {full_name} restored weight differs from saved state (max_diff: {max_diff_saved:.2e})")
    
    result = {
        'total_layers': len(identical_layers) + len(differences),
        'identical_layers': len(identical_layers),
        'different_layers': len(differences),
        'max_diff': max_diff,
        'max_diff_layer': max_diff_layer,
        'differences': differences[:10] if len(differences) > 10 else differences  # 只显示前10个差异
    }
    
    return result


def get_true_shape(tensor, tolerance=1e-6):
    """
    获取张量的真实形状（去除广播维度和重复模式）
    检查每个维度，找到最小重复周期，将相同的值合并在一起
    
    Args:
        tensor: 输入张量
        tolerance: 浮点数比较的容差
    
    Returns:
        tuple: 真实形状（去除广播和重复模式后的形状）
    """
    if tensor.numel() == 0:
        return tensor.shape
    
    true_shape = []
    for dim in range(tensor.ndim):
        dim_size = tensor.shape[dim]
        if dim_size == 1:
            true_shape.append(1)
        else:
            # 提取该维度上的所有切片
            slices = []
            indices = [slice(None)] * tensor.ndim
            for i in range(dim_size):
                indices[dim] = i
                slices.append(tensor[tuple(indices)])
            
            # 检查是否所有值都相同
            all_same = True
            first_slice = slices[0]
            for i in range(1, dim_size):
                if not torch.allclose(first_slice, slices[i], atol=tolerance):
                    all_same = False
                    break
            
            if all_same:
                true_shape.append(1)
            else:
                # 找到最小重复周期
                min_period = dim_size
                for period in range(1, dim_size // 2 + 1):
                    if dim_size % period != 0:
                        continue
                    
                    # 检查是否以 period 为周期重复
                    is_periodic = True
                    num_repeats = dim_size // period
                    for repeat_idx in range(1, num_repeats):
                        for i in range(period):
                            if not torch.allclose(slices[i], slices[repeat_idx * period + i], atol=tolerance):
                                is_periodic = False
                                break
                        if not is_periodic:
                            break
                    
                    if is_periodic:
                        min_period = period
                        break
                
                true_shape.append(min_period)
    
    return tuple(true_shape)



def _load_hf_causal_lm_with_bin_fallback(model_path, **kwargs):
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except FileNotFoundError as exc:
        missing = str(exc)
        if '.safetensors' not in missing:
            raise
        print(
            f'Safetensors checkpoint is incomplete or missing ({missing}); '
            'retrying with PyTorch .bin checkpoint files.'
        )
        kwargs = dict(kwargs)
        kwargs['use_safetensors'] = False
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def reload_model_from_file(model_path, original_model=None):
    """
    从文件重新加载模型，返回一个新的模型对象
    
    参数:
        model_path: 模型路径
        original_model: 可选的原始模型对象（用于获取配置和设备信息）
    
    返回:
        model: 新加载的模型对象，如果失败则返回 None
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    
    print("Reloading model from file to get clean state...")
    try:
        # 重新加载模型
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.use_cache = False
        
        # 确定数据类型和设备
        if original_model is not None:
            # 从原始模型获取数据类型
            dtype = next(original_model.parameters()).dtype
        else:
            dtype = torch.float16
        
        # 创建新的模型对象
        new_model = _load_hf_causal_lm_with_bin_fallback(
            model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        new_model.eval()
        
        print("Model reloaded from file successfully")
        return new_model
    except Exception as e:
        print(f"Warning: Failed to reload model from file: {e}")
        print("Continuing with current model state...")
        return None


def build_model_and_enc(model_path):
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    if vila_10_quant_mode:  
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False}
        )
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
        config.use_cache = False
        if "mpt" in config.__class__.__name__.lower():
            enc = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            enc = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, trust_remote_code=True
            )

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.float16, trust_remote_code=True
            )
        real_quantize_model_weight_0124(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights()

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Older real checkpoints predate GPTQ dynamic-group metadata.  Give
        # them the equivalent static ``k // group_size`` mapping before model
        # dispatch; new checkpoints have this buffer loaded from state_dict.
        from awq.quantize.qmodule import WQLinear0124
        for module in model.modules():
            if isinstance(module, WQLinear0124):
                module.prepare_gptq_group_index()
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)
        prepare_wqlinear0124_cuda_runtime(model)

        
        # p = "/home/zch/project/awq_code/models/CO_REAL/LLAMA1"
        # os.makedirs(p, exist_ok=True)
        # print(f"[SAVE] Saving HF model to: {p}")

        # # 1) 聚合完整 state_dict（适配 device_map/offload）
        # acc = Accelerator()
        # sd = acc.get_state_dict(model)

        # # 2) tokenizer：优先用你已有的 enc，否则从原始基座模型目录加载
        # #    这里假设你外面有 enc（tokenizer），没有就用 args.model_path/args.model
        # try:
        #     tokenizer = enc
        # except NameError:
        #     base_dir = getattr(args, "model_path", None) or getattr(args, "model", None)
        #     tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=True)

        # # 3) 修复 pad_token_id=-1，否则 save_pretrained 会因为 generation_config 校验失败
        # pad_id = getattr(tokenizer, "pad_token_id", None)
        # if pad_id is None or pad_id < 0:
        #     pad_id = getattr(tokenizer, "eos_token_id", None)
        # if pad_id is None or pad_id < 0:
        #     pad_id = 0

        # if getattr(model, "generation_config", None) is not None:
        #     model.generation_config.pad_token_id = int(pad_id)
        # model.config.pad_token_id = int(pad_id)

        # # 4) 直接保存 HF（关键点：把 sd 传进去）
        # model.save_pretrained(
        #     p,
        #     state_dict=sd,
        #     safe_serialization=True,
        #     max_shard_size="2GB",
        # )
        # tokenizer.save_pretrained(p)

        # print("[SAVE] done.")

        # assert 0


        # model.eval()



    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU. When saving a full HuggingFace model, avoid meta tensors
        # left by low-memory loading; save_pretrained needs every parameter to hold data.
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": args.save_model_path is None}
        if args.save_model_path is not None:
            print("save_model_path is set; loading full weights without low_cpu_mem_usage to avoid meta tensors.")
        if not vila_10_quant_mode:
            model = _load_hf_causal_lm_with_bin_fallback(
                model_path, config=config, trust_remote_code=True, **kwargs
            )

        model.eval()

        if args.run_awq:
            assert args.dump_awq, "Please save the awq results with --dump_awq"

            awq_results = run_awq(
                model,
                enc,
                w_bit=args.w_bit,
                q_config=q_config,
                n_samples=128,
                seqlen=512,
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                os.makedirs(dirpath, exist_ok=True)

                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)

            exit(0)

        awq_results = None  # 初始化为 None

        if args.load_awq:

            print("Loading pre-computed AWQ results from", args.load_awq)

            awq_results = torch_load_with_progress(args.load_awq, label="AWQ results", map_location="cpu")
            
            # The real backend only needs per-layer AWQ activation scales.
            # Avoid cloning every FP16 linear weight while a full GPTQ
            # parameter map is also resident on CPU.
            need_awq_x_s = "awq_x_s" not in awq_results
            need_original_weights = (
                args.q_backend == "fake" and "original_weights" not in awq_results
            )
            if need_awq_x_s or need_original_weights:
                from awq.quantize.pre_quant import get_blocks, get_named_linears
                layers = get_blocks(model)
                awq_x_s = {} if need_awq_x_s else awq_results["awq_x_s"]
                original_weights = (
                    {} if need_original_weights else awq_results.get("original_weights")
                )
                for i in range(len(layers)):
                    named_linears = get_named_linears(layers[i])
                    for n, m in named_linears.items():
                        full_name = format_layer_name(model, i, n)
                        if need_awq_x_s:
                            scale_info = find_scale_for_layer(awq_results, full_name)
                            if scale_info is not None:
                                _, _, scales, _ = scale_info
                                if scales is None:
                                    awq_x_s[full_name] = torch.ones(
                                        m.weight.shape[-1], dtype=m.weight.dtype
                                    ).cpu()
                                else:
                                    awq_x_s[full_name] = scales.clone().cpu()
                            else:
                                raise KeyError(f"No AWQ scale found for {full_name}")
                        if need_original_weights:
                            original_weights[full_name] = m.weight.data.clone().cpu()
                if need_awq_x_s:
                    awq_results["awq_x_s"] = awq_x_s
                    print(f"Extracted AWQ activation scales for {len(awq_x_s)} layers")
                if need_original_weights:
                    awq_results["original_weights"] = original_weights
                    print(
                        "Saved original model weights to awq_results, "
                        f"extracted {len(original_weights)} layers"
                    )
            #th_dict = apply_awq722(model, awq_results)
            
            #保存第一次 apply_awq722 后的模型状态，用于后续比较
            model_after_first_awq = None
            if args.w_bit is not None and args.q_backend != "real":
                # 创建一个模型状态的快照用于比较
                from awq.quantize.pre_quant import get_blocks, get_named_linears
                layers = get_blocks(model)
                model_after_first_awq = {}
                for i in range(len(layers)):
                    named_linears = get_named_linears(layers[i])
                    for n, m in named_linears.items():
                        full_name = format_layer_name(model, i, n)
                        model_after_first_awq[full_name] = m.weight.data.clone().cpu()
            
            # 如果需要进行权重量化，需要先对 gptq_weights 应用 scales
            # 方案：保存当前模型状态 -> 重新加载模型 -> 加载 gptq_weights -> 再次 apply_awq722 -> 提取权重 -> 恢复模型
            if args.w_bit is not None and args.mix_sources is None and args.q_backend != "real":
                from awq.quantize.pre_quant import get_blocks, get_named_linears
                
                print("Applying scales to gptq_weights by re-applying AWQ...")
                
                # 1. 保存当前模型状态（apply_awq722 之后的状态）
                model_state_dict = {}
                layers = get_blocks(model)
                for i in range(len(layers)):
                    named_linears = get_named_linears(layers[i])
                    for n, m in named_linears.items():
                        full_name = format_layer_name(model, i, n)
                        # 保存权重到 CPU，同时保存设备信息
                        model_state_dict[full_name] = {
                            'weight': m.weight.data.clone().cpu(),
                            'device': m.weight.device,
                            'dtype': m.weight.dtype
                        }
                
                # 2. 重新加载模型（从文件，得到原始模型状态）
                new_model = reload_model_from_file(args.model_path, original_model=model)
                if new_model is None:
                    print("Failed to reload model, using current model state")
                    new_model = model
                else:
                    # 使用新模型替换当前模型
                    model = new_model
                    layers = get_blocks(model)  # 更新 layers 引用
                
                # 3. 加载 gptq_weights 到模型
                # if 'Llama-7b' in args.model_path:
                #     gptq_weights = torch.load("/home/zch/project/awq_code/models/gptq/gptq_float.pt", map_location="cpu")
                # elif 'llama-2-7b' in args.model_path:
                #     gptq_weights = torch.load("/home/zch/project/awq_code/models/gptq/gptq_float22.pt", map_location="cpu")
                # elif 'Meta-Llama-3-8B' in args.model_path:
                #     gptq_weights = torch.load("/home/zch/project/awq_code/models/gptq/gptq_float23.pt", map_location="cpu")
                # else:
                #     gptq_weights = None

                if 'Llama-7b' in args.model_path:
                    #gptq_weights = torch.load("/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_all_1.pt", map_location="cpu")
                    # gptq_weights = torch.load("/home/zch/project/awq_code/models/gptq_zch/gptq_llama1_pile42_all_s_1.pt", map_location="cpu")
                    # gptq_weights = torch_load_with_progress("/home/zch/project/awq_code/models/gptq3bit/gptq_llama_pilezch.pt", label="GPTQ weights", map_location="cpu")
                    if args.w_bit == 3:
                        gptq_weights = torch_load_with_progress("/home/zch/project/awq_code/models/gptq3bit/gptq_llama_pilezch.pt", label="GPTQ weights", map_location="cpu")
                    else:
                        gptq_weights = torch_load_with_progress("/home/zch/project/awq_code/models/gptq4bit/gptq_llama_pilezch.pt", label="GPTQ weights", map_location="cpu")
                    # gptq_weights = gptq['weight']
                    # gptq_scalses = gptq['scale']
                    # gptq_zeros = gptq['zero']
                elif 'llama-2-7b' in args.model_path:
                    if args.w_bit == 3:
                        gptq_weights = torch_load_with_progress("/home/zch/project/awq_code/models/gptq3bit/gptq_llama2_pilezch.pt", label="GPTQ weights", map_location="cpu")
                    else:
                        gptq_weights = torch_load_with_progress("/home/zch/project/awq_code/models/gptq4bit/gptq_llama2_pilezch.pt", label="GPTQ weights", map_location="cpu")
                    
                    # gptq_weights = gptq['weight']
                    # gptq_scalses = gptq['scale']
                    # gptq_zeros = gptq['zero']
                elif 'Llama-13b' in args.model_path:
                    if args.w_bit == 3:
                        gptq_weights = torch_load_with_progress("/home/zch/project/awq_code/models/gptq3bit/gptq_llama_13b_pilezch.pt", label="GPTQ weights", map_location="cpu")
                    else:
                        gptq_weights = torch_load_with_progress("/home/zch/project/awq_code/models/gptq4bit/gptq_llama_13_pilezch.pt", label="GPTQ weights", map_location="cpu")
                    # gptq_weights = gptq['weight']
                    # gptq_scalses = gptq['scale']
                    # gptq_zeros = gptq['zero']
                elif 'Llama2-13b' in args.model_path:
                    if args.w_bit == 3:
                        gptq_weights = torch_load_with_progress("/home/zch/project/awq_code/models/gptq3bit/gptq_llama2_13b_pilezch.pt", label="GPTQ weights", map_location="cpu")
                    else:
                        gptq_weights = torch_load_with_progress("/home/zch/project/awq_code/models/gptq4bit/gptq_llama2_13b_pilezch.pt", label="GPTQ weights", map_location="cpu")
                elif 'Meta-Llama-3-8B' in args.model_path:
                    gptq_weights = torch_load_with_progress("/home/zch/project/awq_code/models/gptq/gptq_llama1_pile42_all_3.pt", label="GPTQ weights", map_location="cpu")
                    # gptq_weights = gptq['weight']
                    # gptq_scalses = gptq['scale']
                    # gptq_zeros = gptq['zero']
                
                else:
                    gptq_weights = None
                
                if gptq_weights is not None:
                    # 将 gptq_weights 加载到模型（需要先将层移到正确的设备）
                    print("gptq_weight is not none")
                    for i in range(len(layers)):
                        named_linears = get_named_linears(layers[i])
                        for n, m in named_linears.items():
                            full_name = format_layer_name(model, i, n)
                            
                            
                            
                            if full_name in gptq_weights:
                                # 确保层在正确的设备上
                                m.cuda() if torch.cuda.is_available() else None
                                m.weight.data = gptq_weights[full_name]['weight'].to(
                                    device=m.weight.device,
                                    dtype=m.weight.dtype
                                )
                                # 将 scale 广播成和 weight 一样的维度
                                # scale_broadcast = gptq_weights[full_name]['scale'].expand_as(m.weight.data).clone()
                                # m.weight.data = scale_broadcast.to(
                                #     device=m.weight.device,
                                #     dtype=m.weight.dtype
                                # )
                                print("gptq_weights[full_name]['weight'].shape:", gptq_weights[full_name]['weight'].shape)

                                print("gptq_weights[full_name]['scale'].shape:", gptq_weights[full_name]['scale'].shape)
                                print("gptq_weights[full_name]['zero'].shape:", gptq_weights[full_name]['zero'].shape)

                                # expand_scale = gptq_weights[full_name]['scale'].expand_as(m.weight.data).clone()
                                # m.weight.data = expand_scale.to(
                                #     device=m.weight.device,
                                #     dtype=m.weight.dtype
                                # )#expand_zeros = gptq_weights[full_name]['zero'].expand_as(m.weight.data)

                            else:
                                print(f"Layer {full_name} not found in gptq_weights")
                    awq_results["gptq_weights"] = gptq_weights

                    # 4. 再次 apply_awq722（这样 gptq_weights 就会被应用 scales）
                    
                    #apply_awq722(model, awq_results)
                    
                    # 5. 从模型中提取应用了 scales 的权重
                    scaled_gptq_weights = {}
                    for i in range(len(layers)):
                        named_linears = get_named_linears(layers[i])
                        for n, m in named_linears.items():
                            full_name = format_layer_name(model, i, n)
                            if full_name in gptq_weights:
                                # 获取 m.weight.data 的真实形状（去除广播维度）
                                true_shape = get_true_shape(m.weight.data)
                                # print(f"Layer {full_name}:")
                                # print(f"  m.weight.data.shape (broadcasted): {m.weight.data.shape}")
                                # print(f"  m.weight.data true shape (unbroadcasted): {true_shape}")
                                # print(f"  gptq_weights[full_name]['scale'].shape (original): {gptq_weights[full_name]['scale'].shape}")
                                scaled_gptq_weights[full_name] = m.weight.data.clone().cpu()

                    #assert False
                    # 6. 恢复模型到之前的状态
                    # for i in range(len(layers)):
                    #     named_linears = get_named_linears(layers[i])
                    #     for n, m in named_linears.items():
                    #         full_name = format_layer_name(model, i, n)
                    #         if full_name in model_state_dict:
                    #             state = model_state_dict[full_name]
                    #             # 确保层在正确的设备上
                    #             m.cuda() if torch.cuda.is_available() else None
                    #             m.weight.data = state['weight'].to(
                    #                 device=state['device'],
                    #                 dtype=state['dtype']
                    #             )
                    
                    # 7. 将应用了 scales 的 gptq_weights 保存到 awq_results 中，供后续使用
                    if "scaled_gptq_weights" not in awq_results:
                        awq_results["scaled_gptq_weights"] = scaled_gptq_weights
                    print(f"Applied scales to gptq_weights, extracted {len(scaled_gptq_weights)} layers")
                    
                    # 8. 比较恢复后的模型状态与第一次 apply_awq722 后的状态
                    # if model_after_first_awq is not None:
                    #     print("\nComparing model states...")
                    #     comparison_result = compare_model_states(
                    #         model=model,  # 恢复后的模型
                    #         saved_state_dict=model_after_first_awq,  # 第一次 apply_awq722 后的状态
                    #         model_state_dict=model_state_dict  # 保存的状态字典（用于验证）
                    #     )
                        
                    #     if comparison_result['different_layers'] == 0:
                    #         print(f"✓ Model states are identical after restore!")
                    #         print(f"  - Total layers: {comparison_result['total_layers']}")
                    #         print(f"  - Identical layers: {comparison_result['identical_layers']}")
                    #     else:
                    #         print(f"⚠ Found {comparison_result['different_layers']} layers with differences:")
                    #         print(f"  - Total layers: {comparison_result['total_layers']}")
                    #         print(f"  - Identical layers: {comparison_result['identical_layers']}")
                    #         print(f"  - Max difference: {comparison_result['max_diff']:.2e} in layer {comparison_result['max_diff_layer']}")
                    #         print(f"  - First few differences:")
                    #         for diff_info in comparison_result['differences'][:5]:
                    #             print(f"    * {diff_info['layer']}: max_diff={diff_info['max_diff']:.2e}, mean_diff={diff_info['mean_diff']:.2e}")
                    #     print()
                    new_model = reload_model_from_file(args.model_path, original_model=model)
                    if new_model is None:
                        print("Failed to reload model, using current model state")
                        new_model = model
                    else:
                        # 使用新模型替换当前模型
                        model = new_model
                    
                    #orginal_awq_results = torch_load_with_progress(args.load_awq, label="AWQ results", map_location="cpu")

                    # 比较两个 awq_results 是否一致
                    # print("\nComparing awq_results...")
                    # awq_comparison = compare_awq_results(awq_results, orginal_awq_results)
                    
                    # if awq_comparison['is_identical']:
                    #     print(f"✓ awq_results are identical!")
                    #     print(f"  - Total keys: {awq_comparison['total_keys']}")
                    #     print(f"  - Identical keys: {awq_comparison['identical_keys']}")
                    # else:
                    #     print(f"⚠ Found {awq_comparison['different_keys']} different keys in awq_results:")
                    #     print(f"  - Total keys: {awq_comparison['total_keys']}")
                    #     print(f"  - Identical keys: {awq_comparison['identical_keys']}")
                    #     print(f"  - Different keys: {awq_comparison['different_keys']}")
                    #     print(f"  - Differences:")
                    #     for diff_info in awq_comparison['differences'][:10]:  # 只显示前10个差异
                    #         print(f"    * {diff_info['key']}: {diff_info['issue']}")
                    #         if 'max_diff' in diff_info:
                    #             print(f"      (max_diff={diff_info['max_diff']:.2e}, mean_diff={diff_info['mean_diff']:.2e})")
                    #     if len(awq_comparison['differences']) > 10:
                    #         print(f"    ... and {len(awq_comparison['differences']) - 10} more differences")
                    # print()

                    #apply_awq722(model, awq_results)


            # for k in th_dict.keys():
            #     th_dict[k] = 1

            # 导出model
            # save_dir = "/home/zch/project/awq_code/models/llama3"
            # os.makedirs(save_dir, exist_ok=True)

            # print(f"Saving AWQ-quantized model to {save_dir} ...")

            # # ⚠️ 强烈建议先挪到 CPU 再保存
            # model_cpu = model.cpu()
            # torch.cuda.empty_cache()

            # # 保存模型权重 + config
            # model_cpu.save_pretrained(
            #     save_dir,
            #     safe_serialization=True  # 推荐 safetensors
            # )

            # # 保存 tokenizer（一定要！）
            # enc.save_pretrained(save_dir)

            # print("✓ AWQ HuggingFace model saved successfully")
            # print(f"  - config.json")
            # print(f"  - model.safetensors")
            # print(f"  - tokenizer files")

            # # 如果你后面还要继续用 model，可以保持在 CPU
            # model = model_cpu
            # exit(0)


        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert (
                    args.dump_quant is None
                ), "Need to use real quantization to dump quantized weights"
                # 如果加载了 awq_results，传递给量化函数以对 gptq_weights 也应用 scales
                if args.save_model_path is not None:
                    print("save_model_path is set; keeping fake quantization on CPU to avoid meta/offload tensors.")
                pseudo_quantize_model_weight722(
                    model,
                    w_bit=args.w_bit,
                    model_path=args.model_path,
                    quant_type=args.quant_type,
                    q_config=q_config,
                    awq_results=awq_results,
                    mix_sources=args.mix_sources,
                    gptq_params=args.gptq_params,
                    omni_params=args.omni_params,
                    spqr_params=args.spqr_params,
                    keep_on_cpu=args.save_model_path is not None,
                )
                if args.dump_fake:
                    model.save_pretrained(args.dump_fake)
                    print("Pseudo-quantized models saved at", args.dump_fake)
            elif args.q_backend == "real":  # real quantization
                if args.mix_sources is not None:
                    real_sources = {
                        item.strip().lower()
                        for item in args.mix_sources.split(",")
                        if item.strip()
                    }
                    supported_real_sources = (
                        {"awq", "gptq"},
                        {"awq", "rtn"},
                        {"gptq", "rtn"},
                    )
                    if real_sources not in supported_real_sources:
                        raise ValueError(
                            "The real backend supports --mix_sources awq,gptq, "
                            "awq,rtn, or gptq,rtn"
                        )
                real_quantize_model_weight_0124(
                    model,
                    w_bit=args.w_bit,
                    q_config=q_config,
                    awq_results=awq_results,
                    model_path=args.model_path,
                    gptq_params=args.gptq_params,
                    mix_sources=args.mix_sources,
                )
                if args.dump_quant:
                    if not args.dump_quant.endswith("v2.pt"):
                        print("[Info] Auto-change the dump_quant file name to *v2.pt")
                        args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
                    dirpath = os.path.dirname(args.dump_quant)
                    if dirpath:
                        os.makedirs(dirpath, exist_ok=True)

                    print(f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    print("Quantized model saved.")
                    exit(0)


                    # import os, json, shutil
                    # import torch

                    # # ---- 1) 输出目录：把 xxx-v2.pt 改成一个文件夹名 ----
                    # out_dir = args.dump_quant.replace(".pt", "")  # 例如 .../Llama-7b-w4-g128-awq-v2
                    # os.makedirs(out_dir, exist_ok=True)
                    # print(f"[SAVE] Saving HF-format quantized model to: {out_dir}")

                    # # ---- 2)（可选但推荐）把模型放到 CPU 再存，避免 device_map/offload 相关问题 ----
                    # model = model.cpu()
                    # model.eval()

                    # # ---- 3) 修 pad_token_id（你之前也遇到过 generation_config 校验问题）----
                    # pad_id = getattr(tokenizer, "pad_token_id", None)
                    # if pad_id is None or pad_id < 0:
                    #     pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
                    # if getattr(model, "generation_config", None) is not None:
                    #     model.generation_config.pad_token_id = int(pad_id)
                    # model.config.pad_token_id = int(pad_id)

                    # # ---- 4) 保存权重 + config（HF 标准目录）----
                    # # safe_serialization=True -> 写成 safetensors（更安全更通用）
                    # model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
                    # tokenizer.save_pretrained(out_dir)

                    # # ---- 5) 处理“自定义层 WQLinear0124”：把自定义代码打包进目录 ----
                    # # 你说 WQLinear0124 来自 from .qmodule import WQLinear0124
                    # # 这里需要把 qmodule.py 复制到 out_dir，并提供一个 modeling_xxx.py 让 Transformers 能构建同构模型
                    # # 下面两个路径按你项目实际位置改：
                    # shutil.copyfile("/home/zch/project/awq_code/llm-awq_zch0126/awq/quantize/qmodule.py", os.path.join(out_dir, "qmodule.py"))
                    # shutil.copyfile("path/to/your_package/modeling_llama_wqlinear0124.py",
                    #                 os.path.join(out_dir, "modeling_llama_wqlinear0124.py"))

                    # # ---- 6) 更新 config.json：写 auto_map + 量化元信息（让 from_pretrained 能找到自定义类）----
                    # cfg_path = os.path.join(out_dir, "config.json")
                    # cfg = json.load(open(cfg_path, "r"))

                    # cfg["auto_map"] = {
                    #     "AutoModelForCausalLM": "modeling_llama_wqlinear0124.LlamaWQLinear0124ForCausalLM"
                    # }
                    # cfg["quantization_config"] = {
                    #     "quant_method": "wqlinear0124",
                    #     "bits": int(args.w_bit),
                    #     "group_size": int(getattr(q_config, "group_size", 128)),
                    #     "zero_point": bool(getattr(q_config, "zero_point", True)),
                    # }

                    # json.dump(cfg, open(cfg_path, "w"), indent=2)

                    # print("[SAVE] done.")


                    # print(f"Saving the quantized model at {args.dump_quant}...")
                    # torch.save(model.cpu().state_dict(), args.dump_quant)
                    # exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)
        if args.q_backend == "real":
            prepare_wqlinear0124_cuda_runtime(model)

    return model, enc

def load_model():
    """加载AWQ量化模型"""
    # path = "/home/zhangchenhao/project/pretrained_model/Opt"
    path = args.model_path
    try:
        # 如果是.pt文件直接加载
        if path.endswith('.pt'):
            model = torch_load_with_progress(path, label='model .pt', map_location='cpu')
        else:
            # 如果是HuggingFace格式
            model = _load_hf_causal_lm_with_bin_fallback(
                path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

        model.eval()
        print("AWQ模型加载成功")
        return model
    except Exception as e:
        print(f"原模型加载失败: {e}")
        raise



def _save_hf_model(model, enc, save_model_path):
    if save_model_path is None:
        return model
    print(f"保存量化模型到 {save_model_path}...")
    os.makedirs(save_model_path, exist_ok=True)

    model_device = next((p.device for p in model.parameters() if not p.is_meta), None)
    print(f"模型当前设备: {model_device}")

    pad_id = getattr(enc, "pad_token_id", None)
    if pad_id is None or pad_id < 0:
        pad_id = getattr(enc, "eos_token_id", None)
    if pad_id is None or pad_id < 0:
        pad_id = 0
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = int(pad_id)
    model.config.pad_token_id = int(pad_id)

    print("正在聚合模型权重...")
    try:
        state_dict = Accelerator().get_state_dict(model)
    except Exception as exc:
        print(f"Accelerator 聚合 state_dict 失败，改用 model.state_dict(): {exc}")
        state_dict = model.state_dict()

    meta_keys = [k for k, v in state_dict.items() if getattr(v, "is_meta", False)]
    if meta_keys:
        raise RuntimeError(
            "保存失败：state_dict 里仍有 meta tensor，前几个是 "
            f"{meta_keys[:5]}。请关闭 offload/parallel 后重新加载，或使用更大 CPU/GPU 内存。"
        )

    print("正在保存模型权重...")
    model.save_pretrained(
        save_model_path,
        state_dict=state_dict,
        safe_serialization=True,
        max_shard_size="2GB",
    )

    print("正在保存tokenizer...")
    enc.save_pretrained(save_model_path)
    del state_dict
    torch.cuda.empty_cache()

    print(f"✓ HuggingFace 模型已保存到 {save_model_path}")
    print(f"  - 配置文件: {save_model_path}/config.json")
    print(f"  - Tokenizer: {save_model_path}/tokenizer.json 等")
    return model

def _parse_ppl_tasks(tasks):
    if tasks is None:
        return []
    names = []
    for raw in tasks.replace(";", ",").replace(" ", ",").split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name in ("wiki", "wiki2", "wikitext2"):
            name = "wikitext"
        if name not in ("wikitext", "c4"):
            raise ValueError(f"Unsupported PPL task: {raw}. Use wikitext, c4, or wikitext,c4.")
        if name not in names:
            names.append(name)
    return names


def _task_output_path(output_path, task_name, task_names):
    if output_path is None:
        return None
    if len(task_names) <= 1:
        return output_path
    root, ext = os.path.splitext(output_path)
    ext = ext or ".json"
    replacements = [
        ("wikitext_c4", f"both_{task_name}"),
        ("wiki_c4", f"both_{task_name}"),
        ("both", f"both_{task_name}"),
        ("wikitext", task_name),
        ("wiki", task_name),
        ("c4", task_name),
    ]
    for old, new in replacements:
        token = f"_{old}_"
        if token in root:
            return root.replace(token, f"_{new}_") + ext
        if root.endswith(f"_{old}"):
            return root[: -len(old)] + new + ext
    return f"{root}_{task_name}{ext}"


def _write_task_results(output_path, task_name, task_names, results):
    path = _task_output_path(output_path, task_name, task_names)
    if path is None:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {task_name} results to {path}")

def main():
    save_path = "/home/zch/project/awq_code/llm-awq1225/logs"
    # log_file = setup_simple_logging(save_path)
    print(f"Results will be saved to: {save_path}")


    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, will overwrite.")
        # 不再 exit，继续往下执行

    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)
    saved_model_path = None

    task_names = _parse_ppl_tasks(args.tasks)
    if task_names:
        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        # if args.tasks == "wikitext":
        #     # testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        #     print("-------进入if args.tasks == wikitext:--------")
        #     testenc = load_dataset("/home/zhangchenhao/project/dataset/wikitext-2-raw-v1", split="test")
        #     testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
        #     model.seqlen = 2048
        #     testenc = testenc.input_ids.to(model.device)
        #     nsamples = testenc.numel() // model.seqlen
        #     model = model.eval()
        #     nlls = []

        #     # +++
        #     base_model = load_model()
        #     base_model.seqlen = 2048
        #     base_model = base_model.to(model.device)
        #     base_model.eval()
        #     mse_loss = 0
        #     # +++

        #     for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        #         batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
        #             model.device
        #         )

        #         # +++
        #         outputs = model(batch)
        #         base_outputs = base_model(batch)
                
        #         # print('outputs',outputs)
        #         # print('base_outputs', base_outputs)
        #         # assert 0
        #         outputs = outputs[0]
        #         base_outputs = base_outputs[0]
        #         mse_loss += ((base_outputs - outputs).float().pow(2).mean().item())
        #         # +++


        #         with torch.no_grad():
        #             lm_logits = model(batch).logits
        #         shift_logits = lm_logits[:, :-1, :].contiguous().float()
        #         shift_labels = testenc[
        #             :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        #         ][:, 1:]
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(
        #             shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        #         )
        #         neg_log_likelihood = loss.float() * model.seqlen
        #         nlls.append(neg_log_likelihood)

        #     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        #     print(ppl.item())

        #     # +++
        #     print('mse_loss',mse_loss)
        #     # +++

        #     results = {"ppl": ppl.item()}
        #     if args.output_path is not None:
        #         os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        #         with open(args.output_path, "w") as f:
        #             json.dump(results, f, indent=2)
        # else:
        #     task_names = args.tasks.split(",")

        #     lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
        #     results = evaluator.simple_evaluate(
        #         model=lm_eval_model,
        #         tasks=task_names,
        #         batch_size=args.batch_size,
        #         no_cache=True,
        #         num_fewshot=args.num_fewshot,
        #     )

        #     print(evaluator.make_table(results))

        # if args.output_path is not None:
        #     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        #     # otherwise cannot save
        #     results["config"]["model"] = args.model_path
        #     with open(args.output_path, "w") as f:
        #         json.dump(results, f, indent=2)

        if "wikitext" in task_names:
            # testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            print("-------111进入if args.tasks == wikitext:--------")
            testenc = load_dataset("/home/public/dataset/wikitext-2-raw-v1", split="test")
            
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")

            # # 111
            # base_model = load_model()
            # base_model = base_model.to(model.device)
            # base_model.eval()
            # model = base_model
            # # 111 end


            model.seqlen = 2048
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            model = model.eval()
            nlls = []

            # ========== 第一阶段：计算 PPL ==========
            for i in tqdm.tqdm(range(nsamples), desc="evaluating PPL..."):
                start_idx = i * model.seqlen
                end_idx = (i + 1) * model.seqlen
                if end_idx > testenc.shape[1]:
                    break
                batch = testenc[:, start_idx:end_idx].to(model.device)

                with torch.no_grad():
                    outputs = model(batch)
                    lm_logits = outputs.logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = batch[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)
                
                del outputs, lm_logits, shift_logits, shift_labels, loss
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(f"PPL: {ppl.item()}")

            # ========== 第二阶段：计算 MSE ==========
            avg_mse = None
            if args.mseon:
                print("开始计算 MSE...")
                del nlls
                torch.cuda.empty_cache()
                
                try:
                    base_model = load_model()
                    base_model.seqlen = 2048
                    base_model = base_model.cpu()
                    base_model.eval()
                    mse_loss = 0.0
                    mse_count = 0
                    
                    for i in tqdm.tqdm(range(nsamples), desc="evaluating MSE..."):
                        start_idx = i * model.seqlen
                        end_idx = (i + 1) * model.seqlen
                        if end_idx > testenc.shape[1]:
                            break
                        batch = testenc[:, start_idx:end_idx].to(model.device)
                        
                        with torch.no_grad():
                            quant_outputs = model(batch)
                            quant_hidden = quant_outputs[0].cpu()
                            
                            base_model = base_model.to(model.device)
                            base_outputs = base_model(batch)
                            base_hidden = base_outputs[0].cpu()
                            
                            mse_loss += ((base_hidden - quant_hidden).float().pow(2).mean().item())
                            mse_count += 1
                            
                            base_model = base_model.cpu()
                            torch.cuda.empty_cache()
                            del quant_outputs, quant_hidden, base_outputs, base_hidden
                    
                    avg_mse = mse_loss / mse_count if mse_count > 0 else 0.0
                    print(f'MSE Loss: {avg_mse}')
                    del base_model
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Warning: 计算 MSE 时出错: {e}")
                    avg_mse = None

            else:
                print("跳过 MSE 计算（使用 --mseon 参数启用）")  
            
            # ========== 第三阶段：保存量化模型（HuggingFace格式）==========
            # if args.save_model_path is not None:
            #     print(f"保存量化模型到 {args.save_model_path}...")
            #     os.makedirs(args.save_model_path, exist_ok=True)

            #     model_device = next(model.parameters()).device
            #     print(f"模型当前设备: {model_device}")

            #     model_cpu = model.cpu()
            #     torch.cuda.empty_cache()

            #     # ===== 修复 pad_token_id=-1 导致 save_pretrained 失败 =====
            #     pad_id = getattr(enc, "pad_token_id", None)
            #     if pad_id is None or pad_id < 0:
            #         pad_id = getattr(enc, "eos_token_id", None)
            #     if pad_id is None or pad_id < 0:
            #         pad_id = 0
            #     model_cpu.generation_config.pad_token_id = int(pad_id)
            #     model_cpu.config.pad_token_id = int(pad_id)
            #     if getattr(model_cpu.generation_config, "eos_token_id", None) is None and getattr(enc, "eos_token_id", None) is not None:
            #         model_cpu.generation_config.eos_token_id = int(enc.eos_token_id)
            #     if getattr(model_cpu.generation_config, "bos_token_id", None) is None and getattr(enc, "bos_token_id", None) is not None:
            #         model_cpu.generation_config.bos_token_id = int(enc.bos_token_id)
            #     # ===========================================================

            #     print("正在保存模型权重...")
            #     model_cpu.save_pretrained(args.save_model_path, safe_serialization=True)

            #     print("正在保存tokenizer...")
            #     enc.save_pretrained(args.save_model_path)

            #     print(f"✓ 量化模型已保存到 {args.save_model_path}")
            #     model = model_cpu

            results = {"ppl": ppl.item()}
            if avg_mse is not None:
                results["mse_loss"] = avg_mse
                print(f'mse_loss: {avg_mse}')
            _write_task_results(args.output_path, "wikitext", task_names, results)

        if "c4" in task_names:
            from datasets import load_from_disk

            # # 111
            # base_model = load_model()
            # base_model = base_model.to(model.device)
            # base_model.eval()
            # model = base_model
            # # 111 end

            block_size = 2048

            ds = load_from_disk("/home/public/dataset/c4_en_validation")
            # 如果是 DatasetDict，记得 ds = ds["validation"] 或对应 split

            def tok_fn(examples):
                return enc(examples["text"], truncation=False)

            tok = ds.map(tok_fn, batched=True, remove_columns=ds.column_names, num_proc=8)

            # 把 tokens 拼起来再切 block（经典 ppl eval pipeline）
            all_ids = []
            for x in tok["input_ids"]:
                all_ids.extend(x)
            all_ids = torch.tensor(all_ids, dtype=torch.long)

            nsamples = all_ids.numel() // block_size
            nlls = []

            for i in tqdm.tqdm(range(nsamples), desc="evaluating PPL..."):
                batch = all_ids[i*block_size:(i+1)*block_size].unsqueeze(0).to(model.device)
                with torch.no_grad():
                    outputs = model(batch)
                    lm_logits = outputs.logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = batch[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean",
                )
                nlls.append(loss * block_size)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * block_size))
            print("PPL:", ppl.item())

            results = {"ppl": ppl.item()}
            _write_task_results(args.output_path, "c4", task_names, results)

    if args.save_model_path is not None and saved_model_path != args.save_model_path:
        model = _save_hf_model(model, enc, args.save_model_path)

if __name__ == "__main__":
    start_process_gpu_memory_tracking()
    try:
        main()
    finally:
        report_peak_gpu_memory()
