import numpy as np
import torch


# Keep these paths in sync with
# /home/zch/project/awq_code/llm-awq_zch0810mixed-precision/awq/entry.py.
AWQ_WIKITEXT2_PATH = "/home/public/dataset/wikitext-2-raw-v1"
AWQ_C4_VALIDATION_PATH = "/home/public/dataset/c4_en_validation"


def get_wikitext_calibration_tokens(
    model,
    nsamples=32,
    seed=0,
    block_size=2048,
):
    """Sample deterministic full-loss blocks from the local WikiText train split."""
    import random
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if nsamples <= 0:
        raise ValueError("nsamples must be positive")
    dataset = load_dataset(AWQ_WIKITEXT2_PATH, split="train")
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    train_ids = tokenizer(
        "\n\n".join(dataset["text"]), return_tensors="pt"
    ).input_ids
    if train_ids.shape[1] <= block_size:
        raise ValueError("WikiText train tokens are shorter than one block")
    rng = random.Random(seed)
    blocks = []
    for _ in range(nsamples):
        start = rng.randint(0, train_ids.shape[1] - block_size - 1)
        blocks.append(train_ids[:, start : start + block_size])
    return torch.cat(blocks, dim=1), AWQ_WIKITEXT2_PATH


def get_awq_ppl_tokens(name, model, c4_num_proc=8):
    """Build PPL tokens exactly like the AWQ mixed-precision evaluator.

    This intentionally mirrors AWQ's local dataset paths, split, tokenizer
    options, document concatenation, and C4 ``Dataset.map`` pipeline.  It is
    separate from the original SqueezeLLM loaders so paper-reproduction
    behavior remains available.
    """
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )

    if name in ("wikitext", "wikitext2"):
        dataset = load_dataset(AWQ_WIKITEXT2_PATH, split="test")
        tokens = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids
        return tokens, AWQ_WIKITEXT2_PATH

    if name == "c4":
        from tqdm.auto import tqdm

        dataset = load_from_disk(AWQ_C4_VALIDATION_PATH)

        def tokenize(examples):
            return tokenizer(examples["text"], truncation=False)

        tokenized = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=c4_num_proc,
        )
        all_ids = []
        for input_ids in tqdm(
            tokenized["input_ids"],
            desc="Concatenating C4 tokens",
            unit="document",
            dynamic_ncols=True,
        ):
            all_ids.extend(input_ids)
        return (
            torch.tensor(all_ids, dtype=torch.long).unsqueeze(0),
            AWQ_C4_VALIDATION_PATH,
        )

    raise ValueError(f"Unsupported AWQ-compatible PPL dataset: {name}")


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=""):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if "ptb" in name:
        if "new" in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if "c4" in name:
        if "new" in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
