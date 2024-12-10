from datasets import load_dataset

from utils import generate_watermark , detect_watermark , compute_perplexity


import torch
import argparse




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--BASE_MODEL',type=str,default="facebook/opt-1.3b")
    parser.add_argument('--EVAL_MODEL',type=str,default='gpt2-medium')
    parser.add_argument('--message_length',type=int,default=4)
    parser.add_argument('--mb_delta',type=float,default=2.0)
    parser.add_argument('--topk',type=int,default=1000)
    parser.add_argument('--topk_delta',type=float,default=1.0)
    parser.add_argument('--gamma',type=float,default=0.25)
    parser.add_argument('--base',type=int,default=2)
    parser.add_argument('--iter',type=int,default=100)
    parser.add_argument('--max_new_token',type=int,default=200)
    parser.add_argument('--beans',type=int,default=2)
    parser.add_argument('--context_width',type=int,default=1)
    parser.add_argument('--hash_key',type=int,default = 15485863)
    parser.add_argument('--prompt',type=str,default="How is the weather")
    parser.add_argument('--with_topk',action='store_true')



    args = parser.parse_args()
    args = vars(args)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_text_with_topk ,tokenizer, message_binary = generate_watermark(args, args["prompt"], device, with_topk = args["with_topk"])

    print(f"Embedded message:{message_binary}")
    print(f"Output Text:"{output_text})
