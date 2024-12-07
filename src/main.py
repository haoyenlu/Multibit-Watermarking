# Load Model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from tqdm import tqdm



from datasets import load_dataset

from watermark_processor import RepetitionPenaltyLogitsProcessor, MultibitWatermarkLogitsProcessor, TopkLogitsProcessor
from watermark_detector import MultibitWatermarkDetector

import utils
import random
import numpy as np

import torch

import argparse

BASE_MODEL = "facebook/opt-1.3b"
EVAL_MODEL =  'gpt2-medium'


model = AutoModelForCausalLM.from_pretrained(BASE_MODEL , torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")

eval_model = GPT2LMHeadModel.from_pretrained(EVAL_MODEL,device_map="cuda")
eval_tokenizer = GPT2TokenizerFast.from_pretrained(EVAL_MODEL)
eval_model.eval()

with_topk_accuracy = []
with_topk_perplexity = []

without_topk_accuracy = []
without_topk_perplexity = []



def main(args):


    mb_watermark_processor = MultibitWatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.mb_delta,
        base=args.base,
        select_green_tokens=True,
        message_length=args.message_length,
        device=model.device,
    )

    
    min_length_processor = MinLengthLogitsProcessor(
        min_length=1000, 
        eos_token_id=tokenizer.eos_token_id,
        device=model.device
    )

    repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)

    topk_processor = TopkLogitsProcessor(topk=args.topk,delta=args.topk_delta)

    mb_watermark_detector = MultibitWatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        device="cuda" if (torch.cuda.is_available()) else "cpu",
        tokenizer=tokenizer,
        z_threshold=4,
        ignore_repeated_ngrams=False,
        message_length=args.message_length,
        base=args.base,
    )



    c4 = load_dataset("allenai/c4", "en", split='train',streaming=True)

    cnt = 100

    for prompt in c4:
        print(f"Test Sample:{-cnt + 100 + 1}")
        cnt -= 1

        
        input_text = prompt['text']
        tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
        tokenized_input = utils.truncate(tokenized_input, max_length=300).to(model.device)

        message_decimal = random.getrandbits(args.message_length)

        message_binary = format(message_decimal,f"0{args.message_length}b")

        mb_watermark_processor.set_message(message_binary)
        mb_watermark_detector.set_message(message_binary)
        

        """With TopK Logit Processor"""
        output_tokens_with_topK = model.generate(**tokenized_input, max_new_tokens=200, num_beams=2,
                                logits_processor=LogitsProcessorList([
                                    min_length_processor, 
                                    repetition_processor, 
                                    mb_watermark_processor,
                                    topk_processor,
        ]))


        output_text_with_topk = tokenizer.decode(output_tokens_with_topK[0][tokenized_input['input_ids'].shape[-1]:], skip_special_tokens=True) # 生成的文本

        score_dict = mb_watermark_detector.detect(output_text_with_topk, return_scores=True)
        
        with_topk_accuracy.append(score_dict['bit_acc'])

        ppl = utils.compute_perplexity(output_text_with_topk, eval_model, eval_tokenizer)
        with_topk_perplexity.append(ppl)

        print("------With Topk--------")
        print(f"Perplexity:{ppl}")
        print(f"Accuracy:{score_dict['bit_acc']}")

        """Without TopK Logit Processor"""
        output_tokens_without_topK = model.generate(**tokenized_input, max_new_tokens=200, num_beams=2,
                                logits_processor=LogitsProcessorList([
                                    min_length_processor, 
                                    repetition_processor, 
                                    mb_watermark_processor
        ]))


        output_text_without_topk = tokenizer.decode(output_tokens_without_topK[0][tokenized_input['input_ids'].shape[-1]:], skip_special_tokens=True) # 生成的文本

        score_dict = mb_watermark_detector.detect(output_text_without_topk, return_scores=True)
        
        without_topk_accuracy.append(score_dict['bit_acc'])

        ppl = utils.compute_perplexity(output_text_without_topk, eval_model, eval_tokenizer)
        without_topk_perplexity.append(ppl)

        print("------Without Topk--------")
        print(f"Perplexity:{ppl}")
        print(f"Accuracy:{score_dict['bit_acc']}")

    print(f"Average Perplexity with TopK:{sum(with_topk_perplexity)/len(with_topk_perplexity)}")
    print(f"Average Accuracy with Topk:{sum(with_topk_accuracy)/len(with_topk_accuracy)}")

    print(f"Average Perplexity without TopK:{sum(without_topk_perplexity)/len(without_topk_perplexity)}")
    print(f"Average Accuracy without Topk:{sum(without_topk_accuracy)/len(without_topk_accuracy)}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--message_length',type=int,default=4)
    parser.add_argument('--mb_delta',type=float,default=2.0)
    parser.add_argument('--topk',type=int,default=1000)
    parser.add_argument('--topk_delta',type=float,default=1.0)
    parser.add_argument('--gamma',type=float,default=0.25)
    parser.add_argument('--base',type=int,default=2)


    args = parser.parse_args()

    main(args)