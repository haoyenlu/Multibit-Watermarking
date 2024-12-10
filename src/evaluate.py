from datasets import load_dataset

from utils import generate_watermark , detect_watermark , compute_perplexity


import torch
import argparse






def main(args):
    """Analyze Perplexity and Bit Accuracy with and without Topk Logit Processor"""
    with_topk_accuracy = []
    with_topk_perplexity = []

    without_topk_accuracy = []
    without_topk_perplexity = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # dataset
    c4 = load_dataset("allenai/c4", "en", split='train',streaming=True).shuffle()
    c4_iter = iter(c4)

    for _ in range(args["iter"]):
        print(f"Test Sample:{ _ + 1}")
        
        
        input_text = next(c4_iter)['text']

        """With TopK Logit Processor"""
        output_text_with_topk ,tokenizer, message_binary = generate_watermark(args, input_text, device, with_topk= True)
        score_dict_with_topk = detect_watermark(args, output_text_with_topk, message_binary, tokenizer , device)
        with_topk_accuracy.append(score_dict_with_topk['bit_acc'])
        ppl = compute_perplexity(output_text_with_topk, eval_model=args["EVAL_MODEL"], device=device)
        with_topk_perplexity.append(ppl)


        print(f"Perplexity With Top K:{ppl}")
        print(f"Accuracy With Top K:{score_dict_with_topk['bit_acc']}")

        """Without TopK Logit Processor"""
        output_text_without_topk ,tokenizer, message_binary = generate_watermark(args, input_text, with_topk = False)
        score_dict_without_topk = detect_watermark(args, output_text_without_topk, message_binary, tokenizer)
        without_topk_accuracy.append(score_dict_without_topk['bit_acc'])
        ppl = compute_perplexity(output_text_without_topk, eval_model=args["EVAL_MODEL"], device=device)
        without_topk_perplexity.append(ppl)

        print(f"Perplexity Without Top K:{ppl}")
        print(f"Accurac Without Top Ky:{score_dict_without_topk['bit_acc']}")



    print(f"Average Perplexity with TopK:{sum(with_topk_perplexity)/len(with_topk_perplexity)}")
    print(f"Average Accuracy with Topk:{sum(with_topk_accuracy)/len(with_topk_accuracy)}")

    print(f"Average Perplexity without TopK:{sum(without_topk_perplexity)/len(without_topk_perplexity)}")
    print(f"Average Accuracy without Topk:{sum(without_topk_accuracy)/len(without_topk_accuracy)}")





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
    parser.add_argument('--z_threshold',type=float, default=3)


    args = parser.parse_args()

    main(vars(args))