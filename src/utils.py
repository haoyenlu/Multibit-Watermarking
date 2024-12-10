import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

import random


from watermark_processor import MultibitWatermarkLogitsProcessor ,  RepetitionPenaltyLogitsProcessor ,  TopkLogitsProcessor
from watermark_detector import MultibitWatermarkDetector


def truncate(d, max_length=200):
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and len(v.shape) == 2:
            d[k] = v[:, :max_length]
    return d




def compute_perplexity(text, eval_model = 'gpt2-medium', device = 'cpu'):

    eval_model = AutoModelForCausalLM.from_pretrained(eval_model,device_map=device)
    eval_tokenizer = AutoTokenizer.from_pretrained(eval_model)
    eval_model.eval()

    eval_model.eval()
    with torch.no_grad():
        inputs = eval_tokenizer(text, return_tensors='pt', truncation=True, max_length=1024).to(eval_model.device)
        outputs = eval_model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss)

    return perplexity.item()


def generate_watermark(args, input_text , device, with_topk = True):
    
    model = AutoModelForCausalLM.from_pretrained(args["BASE_MODEL"] , torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args["BASE_MODEL"], padding_side="left")

    ### Instantiate logit processor
    mb_watermark_processor = MultibitWatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args["gamma"],
        delta=args["mb_delta"],
        base=args["base"],
        select_green_tokens=True,
        message_length=args["message_length"],
        device=model.device,
        context_width = args["context_width"],
        hash_key=args["hash_key"]
    )

    
    min_length_processor = MinLengthLogitsProcessor(
        min_length=1000, 
        eos_token_id=tokenizer.eos_token_id,
        device=model.device
    )

    repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)

    topK_processor = TopkLogitsProcessor(topk=args["topk"],delta=args["topk_delta"])

    
    ### Generate message
    message_decimal = random.getrandbits(args["message_length"])
    message_binary = format(message_decimal,f"0{args['message_length']}b")

    mb_watermark_processor.set_message(message_binary)

    logit_processor_list = []

    if with_topk:
        logit_processor_list = [min_length_processor, 
                                repetition_processor, 
                                topK_processor, 
                                mb_watermark_processor]
        
    else:
        logit_processor_list = [min_length_processor, 
                                repetition_processor, 
                                mb_watermark_processor]
        
    

    ### Generate text
    tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
    tokenized_input = truncate(tokenized_input, max_length=200).to(model.device)

    output_token = model.generate(**tokenized_input, max_new_tokens= args["max_new_token"], num_beams=args["beans"],
                            logits_processor=LogitsProcessorList(logit_processor_list))

    output_text = tokenizer.decode(output_token[0][tokenized_input['input_ids'].shape[-1]:], skip_special_tokens=True)
    return output_text , tokenizer, message_binary




def detect_watermark(args, input_text , message_binary, tokenizer , device):
    mb_watermark_detector = MultibitWatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args["gamma"],
        device=device,
        tokenizer=tokenizer,
        z_threshold=args["z_threshold"],
        ignore_repeated_ngrams=False,
        message_length=args["message_length"],
        base=args["base"],
        context_width = args["context_width"],
        hash_key=args["hash_key"],
    )


    mb_watermark_detector.set_message(message_binary)

    score_dict = mb_watermark_detector.detect(text=input_text, return_scores=True)

    return score_dict