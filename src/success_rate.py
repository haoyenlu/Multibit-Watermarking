# Load Model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor
)


from datasets import load_dataset

from watermark_processor import RepetitionPenaltyLogitsProcessor, MultibitWatermarkLogitsProcessor, TopkLogitsProcessor
from watermark_detector import MultibitWatermarkDetector

import utils
import random
import numpy as np

import torch

from sklearn.metrics import confusion_matrix

BASE_MODEL = "facebook/opt-1.3b"

def main():
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL , torch_dtype=torch.float16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")

    message_length = 4

    mb_watermark_processor = MultibitWatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        delta=3,
        base=2,
        select_green_tokens=True,
        message_length=message_length,
        device=model.device,
    )

    
    min_length_processor = MinLengthLogitsProcessor(
        min_length=1000, 
        eos_token_id=tokenizer.eos_token_id,
        device=model.device
    )

    repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)

    topk_processor = TopkLogitsProcessor(topk=1000,delta=1)

    mb_watermark_detector = MultibitWatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        device="cuda" if (torch.cuda.is_available()) else "cpu",
        tokenizer=tokenizer,
        z_threshold=1.5,
        ignore_repeated_ngrams=False,
        message_length=message_length,
        base=2,
    )

    
    c4 = load_dataset("allenai/c4", "en", split='train',streaming=True)

    accuracy = []
    total_cnt = 0

    for prompt in c4:
        input_text = prompt['text']
        tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
        tokenized_input = utils.truncate(tokenized_input, max_length=300).to(model.device)

        message_decimal = random.getrandbits(message_length)

        message_binary = format(message_decimal,f"0{message_length}b")
        print("Message:",message_binary)

        mb_watermark_processor.set_message(message_binary)
        mb_watermark_detector.set_message(message_binary)
        
        output_tokens = model.generate(**tokenized_input, max_new_tokens=200, num_beams=2,
                                logits_processor=LogitsProcessorList([
                                    min_length_processor, 
                                    repetition_processor, 
                                    topk_processor,
                                    mb_watermark_processor
        ]))

        output_text = tokenizer.decode(output_tokens[0][tokenized_input['input_ids'].shape[-1]:], skip_special_tokens=True) # 生成的文本

        prefix_and_output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(prefix_and_output_text)

        score_dict = mb_watermark_detector.detect(output_text, return_scores=False)
        print(score_dict)

        tn, fp, fn, tp = confusion_matrix(list(message_binary), list(score_dict['predict_message']))
        accuracy.append(tn + tp)

        total_cnt += 1


    np.save('accuracy.npy',np.array(accuracy))


if __name__ == '__main__':
    main()