import math
import torch
import random


# Load Model
from transformers import LogitsProcessor


def additive_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.sum().item()


class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        base: int = 2,  # base (radix) of each message
        message_length: int = 4,
        code_length: int = 4,
        device: str = "cuda",
        context_width: int = 1,
        hash_key: int = 15485863,
        self_salt: bool = False,
        **kwargs
    ):
        self.device = device
        self.vocab = vocab

        # Watermark behavior:
        self.gamma = gamma
        self.delta = delta
        self.rng = None

        self.context_width = context_width
        self.hash_key = hash_key
        self.self_salt = self_salt


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


        assert math.floor(1 / self.gamma) >= base, f"Only {math.floor(1 / self.gamma)} chunks available " \
                                              f"with current gamma={self.gamma}," \
                                              f"But base is {self.base}"
        self.converted_message = None
        self.message_char = None
        self.bit_position_list = []
        self.position_increment = 0
        self.green_cnt_by_position = {i: [0 for _ in range(self.base)] for i
                                      in range(1, self.converted_msg_length + 1)}

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG.")

        # Generate pseudo random key
        prf_key = additive_prf(input_ids[- self.context_width :], salt_key=self.hash_key)

        position_prf_key = prf_key
        self.prf_key = prf_key

        # seeding for bit position
        random.seed(position_prf_key % (2**64 - 1))
        self.bit_position = random.randint(1, self.converted_msg_length)
        self.message_char = int(self.converted_message[self.bit_position-1])
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_colorlist_ids(self, input_ids: torch.LongTensor, topk: int = None) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)

        _size = topk if topk is not None else len(self.vocab)

        vocab_permutation = torch.randperm(
            _size, device=input_ids.device, generator=self.rng
        )

        colorlist = torch.chunk(vocab_permutation, math.floor(1 / self.gamma))

        return colorlist


    def _get_ngram_score_cached(self, prefix: tuple[int], target: int, cand_msg=None):
        colorlist_ids = self._get_colorlist_ids(torch.as_tensor(prefix, device=self.device))
        colorlist_flag = []

        for cl in colorlist_ids[:self.base]:
            if target in cl:
                colorlist_flag.append(True)
            else:
                colorlist_flag.append(False)

        return colorlist_flag, self.get_current_position()

    def get_current_bit(self):
        return int(self.converted_message[self.bit_position - 1])

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


class MultibitWatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor modifying model output scores in a pipe. Can be used in any HF pipeline to modify scores to fit the watermark,
    but can also be used as a standalone tool inserted for any model producing scores in between model outputs and next token sampler.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""

        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng
        
        for b_idx, input_seq in enumerate(input_ids):
            
            colorlist_ids = self._get_colorlist_ids(input_seq)

            greenlist = colorlist_ids[self.get_current_bit()]

            scores[b_idx][greenlist] += self.delta


        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    changed from huggingface (original version multiple by penalty)
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.
    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    changed based on discussion on github (https://github.com/huggingface/transformers/pull/2303)
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty >= 0):
            raise ValueError(f"`penalty` has to be a non-negasitive float, but is {penalty}")

        self.penalty = penalty
        

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        score = score - self.penalty

        scores.scatter_(1, input_ids, score)

        return scores
    

class TopkLogitsProcessor(LogitsProcessor):
    """Adding bias to topk token"""
    def __init__(self, topk: float, delta: float = 1):

        self.topk = topk

        self.delta = delta

    
    def _cal_topk_mask(self,score , indices):
        mask = torch.zeros_like(score,dtype=torch.bool)
        for idx in indices:
            mask[idx] = True
        
        return mask

    def _get_topk_indices( self, scores: torch.Tensor ):

        topk_score_indices = torch.topk(scores, self.topk, dim=-1).indices

        return topk_score_indices

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        for b_idx , score in enumerate(scores):

            topk_score_indices = self._get_topk_indices(score)
            topk_mask = self._cal_topk_mask(score, topk_score_indices)

            scores[b_idx][topk_mask] += self.delta
            scores[b_idx][~topk_mask] -= self.delta

        return scores
    