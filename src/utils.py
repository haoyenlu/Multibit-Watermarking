import torch


def truncate(d, max_length=200):
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and len(v.shape) == 2:
            d[k] = v[:, :max_length]
    return d

def compute_perplexity(text, eval_model, eval_tokenizer):
    eval_model.eval()
    with torch.no_grad():
        inputs = eval_tokenizer(text, return_tensors='pt', truncation=True, max_length=1024).to(eval_model.device)
        outputs = eval_model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss)

    return perplexity.item()