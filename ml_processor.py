from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    LlamaForCausalLM,
    LlamaTokenizer,
    LogitsProcessorList,
    LogitsProcessor
)

from datasets import load_dataset

import math
import torch
import random


def additive_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.sum().item()




class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        select_green_tokens: bool = True,  # should always be the default if not running in legacy mode
        base: int = 2,  # base (radix) of each message
        message_length: int = 4,
        code_length: int = 4,
        device: str = "cuda",
        **kwargs
    ):
        # patch now that None could now maybe be passed as seeding_scheme
        self.device = device
        # Vocabulary setup
        self.vocab = vocab

        # Watermark behavior:
        self.gamma = gamma
        self.delta = delta
        self.rng = None
        self._initialize_seeding_scheme()
        # Legacy behavior:
        self.select_green_tokens = select_green_tokens

        ### Parameters for multi-bit watermarking ###
        self.original_msg_length = message_length
        self.message_length = max(message_length, code_length)
        decimal = int("1" * message_length, 2)
        self.converted_msg_length = len(self._numberToBase(decimal, base))

        # if message bit width is leq to 2, no need to increase base
        if message_length <= 2: base = 2
        self.message = None
        self.bit_position = None
        self.base = base
        # self.chunk = int(ceil(log2(base)))
        assert math.floor(1 / self.gamma) >= base, f"Only {math.floor(1 / self.gamma)} chunks available " \
                                              f"with current gamma={self.gamma}," \
                                              f"But base is {self.base}"
        self.converted_message = None
        self.message_char = None
        self.bit_position_list = []
        self.position_increment = 0
        self.green_cnt_by_position = {i: [0 for _ in range(self.base)] for i
                                      in range(1, self.converted_msg_length + 1)}

    def _initialize_seeding_scheme(self) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type = "additive_prf"
        self.context_width = 1
        self.self_salt = False
        self.hash_key = 15485863

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(
                f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG."
            )

        prf_key = additive_prf(
            input_ids[-self.context_width :], salt_key=self.hash_key
        )

        position_prf_key = prf_key
        self.prf_key = prf_key

        # seeding for bit position
        random.seed(position_prf_key % (2**64 - 1))
        self.bit_position = random.randint(1, self.converted_msg_length)
        self.message_char = self.get_current_bit(self.bit_position)

        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor, topk: int) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)

        greenlist_size = int(topk * self.gamma)

        vocab_permutation = torch.randperm(
            topk, device=input_ids.device, generator=self.rng
        )

        candidate_greenlist = torch.chunk(vocab_permutation, math.floor(1 / self.gamma))

        return candidate_greenlist[self.message_char]

    def _get_colorlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        colorlist = torch.chunk(vocab_permutation, math.floor(1 / self.gamma))
        return colorlist

    # @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int, cand_msg=None):
        """Expensive re-seeding and sampling is cached."""
        ######################
        # self.converted_message = str(cand_msg) * self.converted_msg_length
        # Handle with care, should ideally reset on __getattribute__ access to self.prf_type, self.context_width, self.self_salt, self.hash_key
        # greenlist_ids = self._get_greenlist_ids(torch.as_tensor(prefix, device=self.device))
        # return True if target in greenlist_ids else False, self.get_current_position()
        ######################
        colorlist_ids = self._get_colorlist_ids(torch.as_tensor(prefix, device=self.device))
        colorlist_flag = []
        for cl in colorlist_ids[:self.base]:
            if target in cl:
                colorlist_flag.append(True)
            else:
                colorlist_flag.append(False)

        return colorlist_flag, self.get_current_position()

    def get_current_bit(self, bit_position):
        # embedding stage
        if self.converted_message:
            return int(self.converted_message[bit_position - 1])
        # extraction stage
        else:
            return 0

    def get_current_position(self):
        return self.bit_position

    def set_message(self, binary_msg: str = ""):
        self.message = binary_msg
        self.converted_message = self._convert_binary_to_base(binary_msg)

    def _convert_binary_to_base(self, binary_msg: str):
        decimal = int(binary_msg, 2)
        converted_msg = self._numberToBase(decimal, self.base)
        converted_msg = "0" * (self.converted_msg_length - len(converted_msg)) + converted_msg
        return converted_msg

    def _numberToBase(self, n, b):
        if n == 0:
            return str(0)
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return "".join(map(str, digits[::-1]))

    def flush_position(self):
        positions = "".join(list(map(str, self.bit_position_list)))
        self.bit_position_list = []
        self.green_cnt_by_position = {i: [0 for _ in range(self.base)] for i
                                      in range(1, self.converted_msg_length + 1)}
        return [positions]


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor modifying model output scores in a pipe. Can be used in any HF pipeline to modify scores to fit the watermark,
    but can also be used as a standalone tool inserted for any model producing scores in between model outputs and next token sampler.
    """

    def __init__(self, top_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = top_k

    def _calc_greenlist_mask(
        self, scores: torch.FloatTensor, greenlist_token_ids
    ) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths

        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    def _bias_greenlist_logits(
        self, scores: torch.Tensor, colorlist_mask: torch.Tensor, greenlist_bias: float
    ) -> torch.Tensor:

        scores[colorlist_mask] = scores[colorlist_mask] + greenlist_bias
        return scores

    def _get_topk_indices( self, scores: torch.Tensor ):

        topk_score_indices = torch.topk(scores, self.top_k, dim=-1).indices
        return topk_score_indices

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""

        # this is lazy to allow us to co-locate on the watermarked model's device
        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng

        #TODO: batchify ecc with feedback
        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length

        for b_idx, input_seq in enumerate(input_ids):
            topk_score_indices = self._get_topk_indices(scores)

            greenlist_ids = self._get_greenlist_ids(input_seq, self.top_k)

            list_of_greenlist_ids[b_idx] = [topk_score_indices[i] for i in greenlist_ids]

            self.bit_position_list.append(self.bit_position)


        green_tokens_mask = self._calc_greenlist_mask(
            scores=scores, greenlist_token_ids=list_of_greenlist_ids
        )
        scores = self._bias_greenlist_logits(
            scores=scores, colorlist_mask=green_tokens_mask,
            greenlist_bias=self.delta
        )

        return scores


def truncate(d, max_length=200):
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and len(v.shape) == 2:
            d[k] = v[:, :max_length]
    return d



if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b" , torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained( "facebook/opt-1.3b", padding_side="left")

        
    watermark_processor = WatermarkLogitsProcessor(
        top_k = 1000,
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,
        delta=2,
        base=2,
        select_green_tokens=True,
        message_length=4,
        device="cuda" if (torch.cuda.is_available()) else "cpu",
    )

    message = '0011'
    watermark_processor.set_message(message)



    # English only
    en = load_dataset("allenai/c4", "en", split='train',streaming=True)

    data_iter = iter(en)

    prompt = next(data_iter)['text']

    tokenized_input = tokenizer(prompt, return_tensors='pt').to(model.device)
    tokenized_input = truncate(tokenized_input, max_length=300)


    print(tokenized_input.shape())

    output_tokens = model.generate(**tokenized_input, max_new_tokens=50, num_beams=4,
                               logits_processor=LogitsProcessorList([watermark_processor]))

    output_text = tokenizer.decode(output_tokens[0][tokenized_input['input_ids'].shape[-1]:], skip_special_tokens=True) # 生成的文本

    prefix_and_output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)


    print(prefix_and_output_text)