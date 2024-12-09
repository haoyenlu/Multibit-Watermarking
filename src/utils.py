import torch

from transformers import LogitsProcessorList

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


def generate(prompt, model, tokenizer, logit_processor_list ,max_new_token=200, beans=2):
    tokenized_input = tokenizer(prompt, return_tensors='pt').to(model.device)
    tokenized_input = truncate(tokenized_input, max_length=300).to(model.device)

    output_token = model.generate(**tokenized_input, max_new_tokens= max_new_token, num_beams=beans,
                            logits_processor=LogitsProcessorList(logit_processor_list))

    output_text = tokenizer.decode(output_token[0][tokenized_input['input_ids'].shape[-1]:], skip_special_tokens=True)
    return output_text

