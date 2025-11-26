from ragas.metrics.base import Metric
from ragas.dataset_schema import SingleTurnSample
import re
import string
from typing import Dict, Any, Optional

class UniversalScorer(Metric):
    name: str = "universal_scorer"
    _required_columns: Dict[str, Any] = {"dataset": "dataset name"}

    def init(self, run_config):
        pass

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks) -> float:
        """
        Async scoring method required by Ragas Metric.
        """
        return self.score(sample)

    def score(self, sample: SingleTurnSample) -> float:
        """
        Dispatch scoring based on dataset name in metadata or sample info.
        We assume 'dataset' is passed in metadata or we infer it.
        For simple-bench, we might pass the dataset name in the sample's metadata.
        """
        # Ragas SingleTurnSample has: user_input, response, reference, retrieved_contexts
        # We map:
        # user_input -> question
        # response -> answer (from agent)
        # reference -> ground_truth
        
        dataset_name = sample.metadata.get("dataset", "").lower()
        prediction = sample.response
        ground_truth = sample.reference

        if "gsm8k" in dataset_name:
            return self._score_gsm8k(prediction, ground_truth)
        elif "gaia" in dataset_name:
            return self._score_gaia(prediction, ground_truth)
        elif "hotpotqa" in dataset_name:
            return self._score_hotpotqa(prediction, ground_truth, dataset_name)
        elif "mmmu" in dataset_name:
            return self._score_mmmu(prediction, ground_truth)
        else:
            # Default to exact match if unknown
            return 1.0 if prediction.strip() == ground_truth.strip() else 0.0

    def _score_gsm8k(self, prediction: str, ground_truth: str) -> float:
        # Extract last number from prediction
        pred_nums = re.findall(r"[-+]?\d*\.\d+|\d+", prediction)
        if not pred_nums:
            return 0.0
        pred_val = float(pred_nums[-1])
        
        # Extract last number from ground truth (usually just a number, but just in case)
        gt_nums = re.findall(r"[-+]?\d*\.\d+|\d+", ground_truth)
        if not gt_nums:
            return 0.0
        gt_val = float(gt_nums[-1])
        
        return 1.0 if abs(pred_val - gt_val) < 1e-6 else 0.0

    def _score_gaia(self, prediction: str, ground_truth: str) -> float:
        def normalize(s):
            return s.lower().strip().translate(str.maketrans('', '', string.punctuation))
        return 1.0 if normalize(prediction) == normalize(ground_truth) else 0.0

    def _score_hotpotqa(self, prediction: str, ground_truth: str, dataset_name: str) -> float:
        # Implement F1 and EM.
        # For the "Custom Accuracy" metric (F1 > 0.8 => 1.0)
        
        def normalize_answer(s):
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)
            def white_space_fix(text):
                return ' '.join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))

        pred_tokens = normalize_answer(prediction).split()
        gt_tokens = normalize_answer(ground_truth).split()
        
        common = set(pred_tokens) & set(gt_tokens)
        num_same = len(common)
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1 = int(pred_tokens == gt_tokens)
        else:
            precision = 1.0 * num_same / len(pred_tokens)
            recall = 1.0 * num_same / len(gt_tokens)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
                
        # Custom Accuracy: If F1 > 0.8, count as success
        return 1.0 if f1 > 0.8 else 0.0

    def _score_mmmu(self, prediction: str, ground_truth: str) -> float:
        # Extract option (A, B, C, D)
        # Simple heuristic: look for "Answer: X" or just "X" at the end.
        # Or just check if the ground truth letter is in the prediction.
        # For strictness, let's try to find the last capital letter that is A-E.
        
        matches = re.findall(r'\b([A-E])\b', prediction.upper())
        if not matches:
            return 0.0
        pred_option = matches[-1]
        
        return 1.0 if pred_option == ground_truth.strip().upper() else 0.0
