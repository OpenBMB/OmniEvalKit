from o_e_Kit.utils.text_normalization.normalization import TextNormalization
from o_e_Kit.utils.text_normalization.en import EnglishTextNormalizer
from o_e_Kit.utils.text_normalization.cn_tn import TextNorm
from o_e_Kit.utils.logger.simple_progress import smart_progress
import re

def split_mixed_text(text):
    """
    Splits mixed English and Chinese text into a list of words and characters.
    e.g., "hi你好" -> ["hi", "你", "好"]
    """
    pattern = r'([a-zA-Z0-9]+|\W+)'
    parts = re.split(pattern, text)
    result = []
    for part in parts:
        if part:
            if re.fullmatch(pattern, part):
                result.append(part.strip())
            else:
                result.extend(list(part.replace(" ", "")))
    return [item for item in result if item]

def get_error_stats(ref, hyp, compute_cer=False):
    """
    Computes word/character error statistics using Levenshtein distance.
    """
    ref_tokens = ref.split() if not compute_cer else list(ref)
    hyp_tokens = hyp.split() if not compute_cer else list(hyp)

    dp = [[0] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    ops = [[(0, 0, 0)] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]

    for i in range(len(ref_tokens) + 1):
        dp[i][0] = i
        ops[i][0] = (0, i, 0)
    for j in range(len(hyp_tokens) + 1):
        dp[0][j] = j
        ops[0][j] = (0, 0, j)

    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            subs_cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            
            sub_err, del_err, ins_err = ops[i - 1][j - 1]
            substitution = (dp[i-1][j-1] + subs_cost, sub_err + subs_cost, del_err, ins_err)
            
            sub_err, del_err, ins_err = ops[i - 1][j]
            deletion = (dp[i-1][j] + 1, sub_err, del_err + 1, ins_err)

            sub_err, del_err, ins_err = ops[i][j-1]
            insertion = (dp[i][j-1] + 1, sub_err, del_err, ins_err + 1)

            dp[i][j], s, d, i_ = min(substitution, deletion, insertion, key=lambda x: x[0])
            ops[i][j] = (s, d, i_)

    _, (subs, dels, ins) = dp[-1][-1], ops[-1][-1]
    return subs, dels, ins, len(ref_tokens)

class WER_Eval:
    """
    Manages the entire WER/CER evaluation process for a dataset.
    This class handles text normalization, error calculation, and summary reporting.
    """
    def __init__(self, lang='en', metric='wer'):
        self.en_norm = EnglishTextNormalizer()
        self.zh_norm = TextNorm(
            to_banjiao=False,
            to_upper=False,
            to_lower=False,
            remove_fillers=False,
            remove_erhua=False,
            check_chars=False,
            remove_space=False,
            cc_mode="",
        )
        self.lang = lang
        self.metric = metric
        self.compute_cer = self.metric == 'cer'
        self.reset()

    def reset(self):
        self.total_subs = 0
        self.total_dels = 0
        self.total_ins = 0
        self.total_words = 0
        self.num_sentences = 0
        self.num_error_sentences = 0
        self.swer_list = []

    def normalize(self, text):
        norm_text = self.en_norm(text)
        if self.lang in ["zh"]:
            norm_text = self.zh_norm(norm_text)
        if self.compute_cer:
            return "".join(split_mixed_text(norm_text))
        else:
            return " ".join(split_mixed_text(norm_text))

    def evaluate(self, predictions):
        self.reset()
        scored_predictions = []
        for item in smart_progress(predictions, desc=f"Calculating {'CER' if self.compute_cer else 'WER'}"):
            ref = item.get('annotation', {}).get('gt_answer', '')
            hyp = item.get('prediction', '').replace("Based on the audio you provided, the transcript of the audio in Engilish is:\n", "")
            
            ref_norm = self.normalize(ref)
            hyp_norm = self.normalize(hyp)

            if not ref_norm:
                continue

            subs, dels, ins, words = get_error_stats(ref_norm, hyp_norm, self.compute_cer)

            self.total_subs += subs
            self.total_dels += dels
            self.total_ins += ins
            self.total_words += words
            self.swer_list.append(float(subs + dels + ins) / words * 100)
            if words > 0:
                self.num_sentences += 1
            
            errors = subs + dels + ins
            if errors > 0 and words > 0:
                self.num_error_sentences += 1
            
            new_item = item.copy()
            new_item['ref_norm'] = ref_norm
            new_item['hyp_norm'] = hyp_norm
            new_item['score'] = round((errors / words) * 100 if words > 0 else 0, 2)
            new_item['details'] = {'subs': subs, 'dels': dels, 'ins': ins, 'words': words}
            scored_predictions.append(new_item)
            
        return scored_predictions

    def summary(self):
        if self.total_words == 0 or self.num_sentences == 0:
            return "Not enough data to generate a summary.", 0.0

        wer = (self.total_subs + self.total_dels + self.total_ins) / self.total_words * 100
        swer = float(sum(self.swer_list)) / len(self.swer_list)
        ser = self.num_error_sentences / self.num_sentences * 100
        
        report = (
            f"\n{'='*20} SUMMARY {'='*20}\n"
            f"Metric:              {'CER' if self.compute_cer else 'WER'}\n"
            f"Sentences:           {self.num_sentences}\n"
            f"Total Words/Chars:   {self.total_words}\n"
            f"Total Errors:        {self.total_subs + self.total_dels + self.total_ins}\n"
            f"  - Substitutions:   {self.total_subs}\n"
            f"  - Deletions:       {self.total_dels}\n"
            f"  - Insertions:      {self.total_ins}\n"
            f"\n"
            f"Error Rate (ERR):    {wer:.2f}%\n"
            f"Error Rate per sample (SWER):{swer:.2f}%\n"
            f"Sentence Error (SER):{ser:.2f}%\n"
            f"{'='*49}\n"
        )
        return report, round(wer, 2)