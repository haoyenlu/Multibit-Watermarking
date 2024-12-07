
import time
import math
import torch
from math import sqrt, ceil, floor, log2, log
import collections
import random

from watermark_processor import WatermarkBase


from itertools import chain, tee


import scipy.stats
from scipy.stats import chisquare, entropy, binom

import numpy as np


from itertools import chain, combinations


def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

class MultibitWatermarkDetector(WatermarkBase):
    """This is the detector for all watermarks imprinted with WatermarkLogitsProcessor.

    The detector needs to be given the exact same settings that were given during text generation  to replicate the watermark
    greenlist generation and so detect the watermark.
    This includes the correct device that was used during text generation, the correct tokenizer, the correct
    seeding_scheme name, and parameters (delta, gamma).

    Optional arguments are
    * normalizers ["unicode", "homoglyphs", "truecase"] -> These can mitigate modifications to generated text that could trip the watermark
    * ignore_repeated_ngrams -> This option changes the detection rules to count every unique ngram only once.
    * z_threshold -> Changing this threshold will change the sensitivity of the detector.
    """

    def __init__(
        self,
        *args,
        device,
        tokenizer,
        z_threshold: float = 1.5,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_ngrams: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        self.ignore_repeated_ngrams = ignore_repeated_ngrams

    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        return_bit_match: bool = True,
        return_z_score_max: bool =True,
        message: str = "",
        **kwargs,
    ):
        s_time = time.time()
        gold_message = self.message

        # sequential extraction 
        ngram_to_watermark_lookup, frequencies_table, ngram_to_position_lookup, green_cnt_by_position, \
            position_list = self._score_ngrams_in_passage_sequential(input_ids)

        # print(green_cnt_by_position)


        # count positions for all tokens
        position_cnt = {}
        for k, v in ngram_to_position_lookup.items():
            freq = frequencies_table[k]
            position_cnt[v] = position_cnt.get(v, 0) + freq

        # compute confidence per position
        p_val_per_position = []
        for p in range(1, self.converted_msg_length + 1):
            all_green_cnt = np.array(green_cnt_by_position[p])
            green_cnt = max(all_green_cnt)
            if position_cnt.get(p) is None:
                position_cnt[p] = 0
            T = position_cnt.get(p)

            # binom_pval = self._compute_binom_p_val(green_cnt, T)
            multi_pval = self._compute_max_multinomial_p_val(green_cnt, T)
            p_val_per_position.append(multi_pval)


        # predict message
        list_decoded_msg, ran_list_decoded_msg, elapsed_time = \
            self._predict_message(position_cnt, green_cnt_by_position, p_val_per_position)

        elapsed_time = elapsed_time - s_time
        best_prediction = list_decoded_msg[0]

        # compute bit accuracy
        correct_bits, total_bits, error_pos = self._compute_ber(best_prediction, gold_message)
        prediction_results = {'confidence': [], 'random': []}

        # for our list decoded msg
        for msg in list_decoded_msg:
            cb, tb, _ = self._compute_ber(msg, gold_message)
            prediction_results['confidence'].append(cb)

        # for random list decoded msg
        for msg in ran_list_decoded_msg:
            cb, tb, _ = self._compute_ber(msg, gold_message)
            prediction_results['random'].append(cb)

        # use the predicted message to select ngram_to_watermark_lookup
        for ngram, green_token in ngram_to_watermark_lookup.items():
            pos = ngram_to_position_lookup[ngram]
            msg = best_prediction[pos - 1]
            ngram_to_watermark_lookup[ngram] = ngram_to_watermark_lookup[ngram][msg]

        green_token_mask, green_unique, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )

        # Count up scores over all ngrams
        if self.ignore_repeated_ngrams:
            position_cnt = {}
            for k, v in ngram_to_position_lookup.items():
                position_cnt[v] = position_cnt.get(v, 0) + 1
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())

        else:
            num_tokens_scored = sum(frequencies_table.values())
            assert num_tokens_scored == len(input_ids) - self.context_width + self.self_salt, "Error for num_tokens_scored"
            green_token_count = sum(
                freq * outcome
                for freq, outcome in zip(
                    frequencies_table.values(), ngram_to_watermark_lookup.values()
                )
            )

        # assert green_token_count == green_token_count_debug, "Debug: green_token_count != green_token_count_debug"

        assert green_token_count == green_unique.sum()

        # HF-style output dictionary
        score_dict = dict()
        sampled_positions = "" if len(position_list) == 0 else "".join(list(map(str, position_list)))
        score_dict.update(dict(sampled_positions=sampled_positions))
        score_dict.update(dict(pred_message="".join(map(str, best_prediction))))
        min_val = min(position_cnt.values())
        max_val = max(position_cnt.values())
        sum_val = sum(position_cnt.values())
        score_dict.update({'min_pos_ratio': min_val / sum_val})
        score_dict.update({'max_pos_ratio': max_val / sum_val})
        score_dict.update(dict(custom_metric=-sum(p_val_per_position)))
        score_dict.update(dict(decoding_time=elapsed_time))
        score_dict.update(dict(confidence_per_position=p_val_per_position))
        score_dict.update(dict(error_pos=error_pos))
        if return_bit_match:
            score_dict.update(dict(bit_acc=correct_bits / total_bits))
            score_dict.update(dict(bit_match=correct_bits == total_bits))
            score_dict.update(dict(cand_match=max(prediction_results['confidence']) == total_bits))
            score_dict.update(dict(cand_match_ablation=max(prediction_results['random']) == total_bits))
            score_dict.update(dict(cand_acc=max(prediction_results['confidence']) / total_bits))
            score_dict.update(dict(cand_acc_2=max(prediction_results['confidence'][:3]) / total_bits))
            score_dict.update(dict(cand_acc_4=max(prediction_results['confidence'][:5]) / total_bits))
            score_dict.update(dict(cand_acc_8=max(prediction_results['confidence'][:9]) / total_bits))
            score_dict.update(dict(cand_acc_ablation=max(prediction_results['random']) / total_bits))

        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(
                dict(z_score=z_score)
            )
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask.tolist()))
        if return_z_at_T:
            # Score z_at_T separately:
            sizes = torch.arange(1, len(green_unique) + 1)
            seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
            seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
            z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
            z_score_at_T = z_score_at_effective_T[offsets]
            assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))
            score_dict.update(dict(z_score_at_T=z_score_at_T))

        return score_dict

    def _compute_z_score(self, observed_count, T):
        """
        count refers to number of green tokens, T is total number of tokens
        If T <= 0, this means this position was not sampled. Return 0 for this.
        """
        if T <= 0:
            return 0
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = math.sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_binom_p_val(self, observed_count, T):
        """
        count refers to number of green tokens, T is total number of tokens
        If T <= 0, this means this position was not sampled. Return p-val=1 for this.
        """
        if T <= 0:
            return 1
        binom_p = self.gamma
        observed_count -= 1
        # p value for observing a sample geq than the observed count
        p_val = 1 - binom.cdf(max(observed_count, 0), T, binom_p)
        return p_val

    def _compute_max_multinomial_p_val(self, observed_count, T):
        """
        Compute the p-value by subtracting the cdf(observed_count -1) of multinomial~(T, 1/base, ... 1/base),
        which is the probability of observing a sample as extreme or more as the observed_count
        The computation follows from Levin, Bruce. "A representation for multinomial cumulative distribution functions."
        The Annals of Statistics (1981): 1123-1126.
        """
        if T <= 0:
            return 1
        poiss = scipy.stats.poisson
        normal = scipy.stats.norm
        k = self.base
        s = T
        a = observed_count - 1
        poiss_cdf_X = poiss.cdf(a, T / k)
        normal_approx_W = normal.cdf(0.5 / np.sqrt(T)) - normal.cdf(-0.5 / np.sqrt(T))
        log_max_multi_cdf = math.log(np.sqrt(2 * math.pi * T)) + k * math.log(poiss_cdf_X) + math.log(normal_approx_W)
        max_multi_cdf = math.exp(log_max_multi_cdf)
        p_val = 1 - min(1, max_multi_cdf)
        return p_val

    def _compute_hoeffdings_bound(self, observed_count, T):
        """
        Compute bound using Hoeffding's inequality.
        Similar to using the normal approximation to the binomial
        """
        if T <= 0:
            return 1
        mean = T / self.base
        delta = max(0, observed_count - mean)
        bound = math.exp(-2 * delta ** 2 / T)
        return bound


    def _predict_message(self, position_cnt, green_cnt_by_position, p_val_per_pos,
                         num_candidates=16):
        s_time = 0
        msg_prediction = []
        confidence_per_pos = []
        for pos in range(1, self.converted_msg_length + 1):
            # find the index (digit) with the max counts of colorlist
            p_val = p_val_per_pos[pos-1]
            if position_cnt.get(pos) is None: # no allocated tokens (may happen when T / b is small)
                position_cnt[pos] = -1
                preds = random.sample(list(range(self.base)), 2)
                pred = preds[0]
                next_idx = preds[1]
                confidence_per_pos.append((-p_val, pred, next_idx, pos))

            else:
                green_counts = green_cnt_by_position[pos]
                pred, val = max(enumerate(green_counts), key=lambda x: (x[1], x[0]))
                sorted_idx = np.argsort(green_counts)
                max_idx, next_idx = sorted_idx[-1], sorted_idx[-2]
                confidence_per_pos.append((-p_val, max_idx, next_idx, pos))
            msg_prediction.append(pred)

        elapsed_time = time.time() - s_time
        random_prediction_list = [msg_prediction]

        # sample random bits
        cnt = 0
        while cnt < num_candidates:
            msg_decimal = random.getrandbits(self.message_length)
            converted_msg = self._numberToBase(msg_decimal, self.base)
            converted_msg = "0" * (self.converted_msg_length - len(converted_msg)) + converted_msg
            random_prediction_list.append(converted_msg)
            cnt += 1

        msg_prediction_list = [msg_prediction]
        num_candidate_position = ceil(log2(num_candidates + 1))

        # sort by the least confident positions
        confidence_per_pos = sorted(confidence_per_pos, key=lambda x: x[0])[:num_candidate_position]
        cnt = 0
        candidate_iter = iter(powerset(confidence_per_pos))
        while cnt < num_candidates:
            try:
                candidate = next(candidate_iter)
            except:
                break
            cand_msg = msg_prediction.copy()
            for _, max_idx, next_idx, pos in candidate:
                cand_msg[pos - 1] = next_idx
            msg_prediction_list.append(cand_msg)
            cnt += 1

        return msg_prediction_list, random_prediction_list, elapsed_time

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _score_ngrams_in_passage_sequential(self, input_ids: torch.Tensor):
        position_list = [] # list of bit position
        frequencies_table = {}
        ngram_to_position_lookup = {}
        ngram_to_watermark_lookup = {}

        green_cnt_by_position = {i: [0 for _ in range(self.base)] for i in range(1, self.converted_msg_length + 1)}
        increment = self.context_width - self.self_salt

        # loop through tokens to get the sampled positions
        for idx in range(self.context_width, len(input_ids) + self.self_salt):
            pos = increment % self.converted_msg_length + 1
            ngram = input_ids[idx - self.context_width: idx + 1 - self.self_salt]
            ngram = tuple(ngram.tolist())
            frequencies_table[ngram] = frequencies_table.get(ngram, 0) + 1
            target = ngram[-1]
            prefix = ngram if self.self_salt else ngram[:-1]

            colorlist_flag, pos = self._get_ngram_score_cached(prefix, target)

            for _idx, flag in enumerate(colorlist_flag):
                if flag:
                    green_cnt_by_position[pos][_idx] += 1

            ngram_to_watermark_lookup[ngram] = colorlist_flag
            position_list.append(pos)
            ngram_to_position_lookup[ngram] = pos
            increment += 1

        return ngram_to_watermark_lookup, frequencies_table, ngram_to_position_lookup, green_cnt_by_position, \
            position_list



    def _get_green_at_T_booleans(self, input_ids, ngram_to_watermark_lookup) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate binary list of green vs. red per token, a separate list that ignores repeated ngrams, and a list of offsets to
        convert between both representations:
        green_token_mask = green_token_mask_unique[offsets] except for all locations where otherwise a repeat would be counted
        """
        green_token_mask, green_token_mask_unique, offsets = [], [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt)

        for idx, ngram_example in enumerate(ngram_examples):
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example in used_ngrams:
                    pass
                else:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(unique_ngram_idx - 1)
        return (torch.tensor(green_token_mask),
                torch.tensor(green_token_mask_unique),
                torch.tensor(offsets),
                )

    def _score_windows_impl_batched(
        self,
        input_ids: torch.Tensor,
        window_size: str,
        window_stride: int = 1,
    ):
        # Implementation details:
        # 1) --ignore_repeated_ngrams is applied globally, and windowing is then applied over the reduced binary vector
        #      this is only one way of doing it, another would be to ignore bigrams within each window (maybe harder to parallelize that)
        # 2) These windows on the binary vector of green/red hits, independent of context_width, in contrast to Kezhi's first implementation
        # 3) z-scores from this implementation cannot be directly converted to p-values, and should only be used as labels for a
        #    ROC chart that calibrates to a chosen FPR. Due, to windowing, the multiple hypotheses will increase scores across the board#
        #    naive_count_correction=True is a partial remedy to this

        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(input_ids)
        green_mask, green_ids, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )
        len_full_context = len(green_ids)

        partial_sum_id_table = torch.cumsum(green_ids, dim=0)

        if window_size == "max":
            # could start later, small window sizes cannot generate enough power
            # more principled: solve (T * Spike_Entropy - g * T) / sqrt(T * g * (1 - g)) = z_thresh for T
            sizes = range(1, len_full_context)
        else:
            sizes = [int(x) for x in window_size.split(",") if len(x) > 0]

        z_score_max_per_window = torch.zeros(len(sizes))
        cumulative_eff_z_score = torch.zeros(len_full_context)
        s = window_stride

        window_fits = False
        for idx, size in enumerate(sizes):
            if size <= len_full_context:
                # Compute hits within window for all positions in parallel:
                window_score = torch.zeros(len_full_context - size + 1, dtype=torch.long)
                # Include 0-th window
                window_score[0] = partial_sum_id_table[size - 1]
                # All other windows from the 1st:
                window_score[1:] = partial_sum_id_table[size::s] - partial_sum_id_table[:-size:s]

                # Now compute batched z_scores
                batched_z_score_enum = window_score - self.gamma * size
                z_score_denom = sqrt(size * self.gamma * (1 - self.gamma))
                batched_z_score = batched_z_score_enum / z_score_denom

                # And find the maximal hit
                maximal_z_score = batched_z_score.max()
                z_score_max_per_window[idx] = maximal_z_score

                z_score_at_effective_T = torch.cummax(batched_z_score, dim=0)[0]
                cumulative_eff_z_score[size::s] = torch.maximum(
                    cumulative_eff_z_score[size::s], z_score_at_effective_T[:-1]
                )
                window_fits = True  # successful computation for any window in sizes

        if not window_fits:
            raise ValueError(
                f"Could not find a fitting window with window sizes {window_size} for (effective) context length {len_full_context}."
            )

        # Compute optimal window size and z-score
        cumulative_z_score = cumulative_eff_z_score[offsets]
        optimal_z, optimal_window_size_idx = z_score_max_per_window.max(dim=0)
        optimal_window_size = sizes[optimal_window_size_idx]
        return (
            optimal_z,
            optimal_window_size,
            z_score_max_per_window,
            cumulative_z_score,
            green_mask,
        )

    def _score_sequence_window(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        window_size: str = None,
        window_stride: int = 1,
    ):
        (
            optimal_z,
            optimal_window_size,
            _,
            z_score_at_T,
            green_mask,
        ) = self._score_windows_impl_batched(input_ids, window_size, window_stride)

        # HF-style output dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=optimal_window_size))

        denom = sqrt(optimal_window_size * self.gamma * (1 - self.gamma))
        green_token_count = int(optimal_z * denom + self.gamma * optimal_window_size)
        green_fraction = green_token_count / optimal_window_size
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=green_fraction))
        if return_z_score:
            score_dict.update(dict(z_score=optimal_z))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=z_score_at_T))
        if return_p_value:
            z_score = score_dict.get("z_score", optimal_z)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))

        # Return per-token results for mask. This is still the same, just scored by windows
        # todo would be to mark the actually counted tokens differently
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_mask.tolist()))

        return score_dict

    def _compute_ber(self, pred_msg: list, message: str):
        pred_msg = "".join(map(str, pred_msg))
        decimal = int(pred_msg, self.base)
        decimal = min(decimal, 2 ** self.message_length - 1)
        binary_pred = format(decimal, f"0{self.message_length}b")

        # predicted binary message is longer because the last chunk was right-padded
        if len(binary_pred) != len(message):
            print(f"Predicted msg: {pred_msg}")
            print(f"Predicted binary msg: {binary_pred}")
            print(f"Gold msg: {message}")
            raise RuntimeError("Extracted message length is shorter the original message!")

        _match = 0
        _total = 0
        error_pos = []
        for pos, (g, p) in enumerate(zip(message, binary_pred)):
            if g == p:
                _match += 1
                error_pos.append(False)
            else:
                error_pos.append(True)
            _total += 1
        return _match, _total, error_pos

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        convert_to_float: bool = False,
        **kwargs,
    ) -> dict:
        """Scores a given string of text and returns a dictionary of results."""
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"

        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )

            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)

            # Remove bos-tok

        if  tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]

        # call score method
        output_dict = {}

        # Using Window Detection
        if window_size is not None:
            # assert window_size <= len(tokenized_text) cannot assert for all new types
            score_dict = self._score_sequence_window(
                tokenized_text,
                window_size=window_size,
                window_stride=window_stride,
                **kwargs,
            )
            output_dict.update(score_dict)
        else:
            kwargs['text'] = text
            score_dict = self._score_sequence(tokenized_text, **kwargs)

        if return_scores:
            output_dict.update(score_dict)
            # score sampled positions
            gold_position = kwargs['position'][self.context_width - self.self_salt:]
            position = score_dict['sampled_positions']
            match_cnt = sum([x == y for x, y in zip(gold_position, position)])
            output_dict.update(dict(position_acc=match_cnt / len(position)))

        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert (z_threshold is not None), "Need a threshold in order to decide outcome of detection test"
            output_dict['predict_message'] = score_dict['pred_message']
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            output_dict['z_score'] = score_dict['z_score']
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        # convert any numerical values to float if requested
        if convert_to_float:
            for key, value in output_dict.items():
                if isinstance(value, int):
                    output_dict[key] = float(value)

        return output_dict