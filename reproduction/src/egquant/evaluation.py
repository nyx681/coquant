import torch

@torch.no_grad()
def perplexity(model, tokens, device, block_size=2048, batch_size=1):
    tokens = tokens.reshape(1, -1)
    count = tokens.numel() // block_size
    if count == 0 or batch_size < 1:
        raise ValueError('Need at least one full block and a positive batch size')
    previous = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    nlls = []
    try:
        for start in range(0, count, batch_size):
            n = min(batch_size, count - start)
            batch = tokens[:, start * block_size:(start + n) * block_size].reshape(n, block_size).to(device)
            logits = model(batch).logits
            shift_logits = logits[:, :-1].contiguous().float()
            loss = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), batch[:, 1:].contiguous().reshape(-1), reduction='mean')
            nlls.append(loss.float() * (n * block_size))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite evaluation loss')
            if (start + n) % (10 * batch_size) == 0:
                print(f'Evaluated {start + n}/{count} blocks', flush=True)
        result = float(torch.exp(torch.stack(nlls).sum() / (count * block_size)))
    finally:
        model.config.use_cache = previous
    return result
