## From-scratch Quantization

Here we provide the codes for quantizing your custom models from scratch. Follow the steps outlined below.

**NOTE:** Only dense-only quantization is supported now. We are working to release the code for the Dense-and-Sparse quantization soon.

### 0. Prerequisite

In addition to installing the dependencies required for the inference code, you will need to install additional dependencies by running the following command:
```
conda activate sqllm
pip install scikit-learn==1.3.1
```
Additionally, make sure you have your own LLaMA Huggingface checkpoint saved at `[MODEL_PATH]`.


### 1. Compute gradients (Fisher-based sensitivity score)
SqueezeLLM employs the Fisher Information matrix as a sensitivity metric.
To compute this, we offer a separate [separate framework](https://github.com/kssteven418/SqueezeLLM-gradients) where you can compute the gradient square for your target model. 
This framework will produce the gradient square in the same format as the original Huggingface model checkpoint for your target model, with the only difference being that the weight values are replaced by the gradient square.

### 2. Chunk model weights and gradients
You should now have the model checkpoint at `[MODEL_PATH]` and the gradient checkpoint computed in the previous step at `[GRADIENT_PATH]`. 
Our framework requires that both checkpoints are chunked at the layer granularity to reduce the model loading overhead. 
Run the following code to chunk both your model and gradient checkpoints:
```
python chunk_models.py --model [MODEL_PATH] --output [MODEL_CHUNKS_PATH] --model_type llama
python chunk_models.py --model [GRADIENT_PATH] --output [GRADIENT_CHUNKS_PATH] --model_type llama
```

This will save model weights and gradients in the layer granularity as `[MODEL_CHUNKS_PATH]` and `[GRADIENT_CHUNKS_PATH]`.

### 3. (Optional for D+S quantization) Outlier configuration generation
This is an optional step to generate an outlier configuration for the Dense-and-Sparse quantization.
This step creates a configuration file that defines thresholds for identifying outlier values in the weights.
Note that while SqueezeLLM extracts both outliers and sensitive values to sparse matrices, 
there's no need for a separate configuration step for sensitive values -- this is covered in step 4.
This step only generates a configuration of outlier values.

Run the following command to generate an outlier configuration:
```
python generate_outlier_config.py --model [MODEL_CHUNKS_PATH] --range [RANGE] --output [OUTLIERS_CONFIG_PATH]
```

* `--model`: Points to the chunked model weights obtained in the previous step
* `--range`: Determines the thresholds T_min and T_max as described in Section 4.2 of the paper. Adjusting this value changes the absolute values of T_min and T_max, consequently affecting the total number of outliers. A larger range value decreases the number of outliers.
* `--output`: The path where the outlier configuration will be saved, formatted as `[OUTLIERS_CONFIG_PATH]/outlier_config_o{outlier_percentage}.json`, where outlier_percentage represents the percentage of values classified as outliers. You will need to fine-tune the `range` argument to achieve a desired outlier percentage, with 1.5-2.0 being a recommended starting range.



### 4. K-means clustering
Run the following code to perform K-means clustering, which will yield the non-uniform quantization look-up table (LUT):
```
python nuq.py --bit 4 --model_type llama --model [MODEL_CHUNKS_PATH] --gradient [GRADIENT_CHUNKS_PATH] --output [LUT_PATH]
```
The `--bit` argument is the bit-precision, and can be set to either 3 or 4. 
The `--model` and `--gradient` arguments should point to the chunked model weights and gradients obtained in the previous step. 
The resulting LUT entries will be stored in `[LUT_PATH]/lut`.

To only quantize a specific range of layers, you can use the `--range` option. For instance, assigning `--range 0,10` will only compute LUT entries for layers 0 to 9.

Please note that this process is highly CPU-intensive, so it is recommended to run the code in environments with multiple and stronger CPU cores for faster computation.

### AWQ-delta weighted K-means

For the repository's local Llama-7B W4/G128 paths, the complete export,
chunking, LUT generation, fake dequantization, and WikiText2+C4 PPL workflow is
available as a resumable one-command script:

```bash
bash /home/zch/project/Quantification/SqueezeLLM/run_awq_delta_ppl.sh
```

As an alternative to Fisher weighting, `nuq.py` can use the element-wise error
of an AWQ fake-quantized/dequantized model:

```text
delta_W = abs(W_original - W_AWQ_dequantized)
objective = sum(delta_W * (W_original - center[label]) ** 2)
```

`W_original` remains the data being clustered. The AWQ-dequantized checkpoint
is used only to construct K-means `sample_weight`; its weights are not copied
into the resulting SqueezeLLM model.

For the local AWQ fork, an AWQ-only dequantized checkpoint can be exported with
`--dump_fake` (run this command from that AWQ repository and use its matching
environment):

```bash
python -m awq.entry \
  --model_path [ORIGINAL_MODEL_PATH] \
  --w_bit 4 \
  --q_group_size 128 \
  --q_backend fake \
  --mix_sources awq \
  --load_awq [AWQ_SEARCH_RESULTS_PT] \
  --dump_fake [AWQ_DEQUANTIZED_MODEL_PATH]
```

First chunk both the original model and the AWQ-dequantized Hugging Face model.
The latter must contain ordinary floating-point Linear weights after AWQ fake
quantization/dequantization, with the same architecture as the original model:

```bash
python chunk_models.py \
  --model [ORIGINAL_MODEL_PATH] \
  --model_type llama \
  --output_path [ORIGINAL_MODEL_CHUNKS]

python chunk_models.py \
  --model [AWQ_DEQUANTIZED_MODEL_PATH] \
  --model_type llama \
  --output_path [AWQ_DEQUANTIZED_CHUNKS]
```

Generate a dense-only 3-bit or 4-bit SqueezeLLM LUT using AWQ delta instead of
Fisher. `--gradient` is not needed in this mode:

```bash
python nuq.py \
  --bit 4 \
  --model_type llama \
  --model [ORIGINAL_MODEL_CHUNKS] \
  --awq_model [AWQ_DEQUANTIZED_CHUNKS] \
  --output_folder [AWQ_DELTA_LUT_PATH]
```

The run writes `kmeans_config.json` with `weight_source: "awq_delta"`. Every
AWQ tensor is checked against the corresponding original tensor. If one row's
delta values are all zero, that row falls back to ordinary equal-weight
K-means.

Evaluate the resulting LUT as fake quantization without packing it:

```bash
CUDA_VISIBLE_DEVICES=0 python ../llama.py \
  [ORIGINAL_MODEL_PATH] wikitext2 \
  --wbits 4 \
  --fake_lut_folder [AWQ_DELTA_LUT_PATH] \
  --awq_fair_eval \
  --awq_fair_datasets wikitext,c4 \
  --ppl_output [PPL_RESULT_JSON]
```

Do not pass `--include_sparse` for this dense-only experiment. `--gradient`
may still be supplied together with `--awq_model` only when Fisher-based
`--sensitivity` extraction is also requested; it does not replace AWQ delta as
the K-means weight source.

**Additional arguments for D+S quantization:**
To perform Dense-and-Sparse quantization, e.g. with 0.45% outliers and 0.05% sensitive values as in the paper, you will first need to generate an outlier configuration file for 0.45% outliers by completing step 3. This file will be stored as `[OUTLIERS_CONFIG_PATH]/outlier_config_o0.45.json`.
Then, run the following command:
```
python nuq.py --bit 4 --model_type llama --model [MODEL_CHUNKS_PATH] --gradient [GRADIENT_CHUNKS_PATH] --output [LUT_PATH] --outlier_config [OUTLIERS_CONFIG_PATH]/outlier_config_o0.45.json --sensitivity 0.05
```
* `--outlier_config`: reads the outlier configuration file generated in the previous step to extract 0.45% of the outliers to the sparse matrices accordingly.
* `--sensitivity`: takes out an additional 0.05% of sensitive values to the sparse matrices. 



### 4. Packing
Finally, use the obtained LUT from the previous step to save your model into a packed format. Run the following command:
```
python pack.py --model [MODEL_PATH] --wbits 4 --folder [LUT_PATH] --save [PACKED_CKPT_PATH]
```
`[MODEL_PATH]` is the original model checkpoint, and `[LUT_PATH]` is the location where the LUT is stored from the previous step. 
The packed checkpoint will be saved at `[PACKED_CKPT_PATH]`, which can now be immediately used in your inference code.

**Additional arguments for D+S packing:**
When you performed D+S quantization in the previous step, you will obtain the outliers file as well as LUT in `[LUT_PATH]`.
In this case, you will need to  proceed with D+S packing using the following command:
```
python pack.py --model [MODEL_PATH] --wbits 4 --folder [LUT_PATH] --save [PACKED_CKPT_PATH] --include_sparse --balance
```
This command incorporates two additional arguments, `--include_sparse` and `--balance`.

## External per-weight importance masks

`nuq.py` can replace SqueezeLLM's built-in Fisher/outlier sparse selector with
an externally generated per-Linear mask. The supported artifact format is
`squeezellm_external_importance_mask_v1`, generated by the AWQ mixed-precision
fork's `--dump_importance_mask` option.

Run the 4-bit LUT generation with the external mask:

```bash
python nuq.py \
  --bit 4 \
  --model_type llama \
  --model [MODEL_CHUNKS_PATH] \
  --gradient [GRADIENT_CHUNKS_PATH] \
  --output_folder [LUT_PATH] \
  --importance_mask [IMPORTANCE_MASK_PATH]
```

The selected weights are removed from the dense matrix and saved as an FP32
sparse matrix. The remaining weights use SqueezeLLM's Fisher-weighted 4-bit
non-uniform K-means/LUT quantization. `--importance_mask` can be combined with
`--outlier_config` to keep the original magnitude-outlier component fixed, but
cannot be combined with `--sensitivity` in the same run.

Pack the result as a real INT4 plus sparse-FP32 checkpoint:

```bash
python pack.py \
  --model [MODEL_PATH] \
  --wbits 4 \
  --folder [LUT_PATH] \
  --save [PACKED_CKPT_PATH] \
  --include_sparse \
  --balance
```

The mask and chunked model must describe the same architecture. Every target
Linear matrix is shape-checked before quantization.

### AWQ-compatible perplexity comparison

For a fair comparison with
`/home/zch/project/awq_code/llm-awq_zch0810mixed-precision`, use the dedicated
evaluation mode in `llama.py` after packing:

```bash
CUDA_VISIBLE_DEVICES=0 python llama.py \
  /mnt/dfq/pretrained_model/Llama-7b \
  wikitext2 \
  --wbits 4 \
  --load /mnt/dfq/squeezellm_mixed/packed/Llama-7b-int4-fp32-1pct.pt \
  --include_sparse \
  --awq_fair_eval \
  --awq_fair_datasets wikitext \
  --ppl_output /mnt/dfq/squeezellm_mixed/results/Llama-7b-int4-fp32-1pct-ppl.json
```

This command evaluates WikiText2 only. It intentionally uses the same local
WikiText2 path as the AWQ fork:

```text
/home/public/dataset/wikitext-2-raw-v1
```

It also matches AWQ's test split, slow tokenizer, document concatenation,
2048-token blocks, and cross-entropy aggregation. Omit
`--awq_fair_datasets wikitext` to use the default full WikiText2+C4 fair
evaluation. The positional `wikitext2` argument is retained because it is
required by the original `llama.py` command-line interface.
