from typing import List, Dict
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def evaluate_answers(preds: List[str], refs: List[str]) -> Dict[str, float]:
    rs = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1, rl = 0.0, 0.0
    for p, r in zip(preds, refs):
        s = rs.score(r, p)
        r1 += s["rouge1"].fmeasure
        rl += s["rougeL"].fmeasure
    P, R, F1 = bert_score(preds, refs, lang="en", rescale_with_baseline=True)
    n = max(1, len(preds))
    return {"rouge1_f": r1/n, "rougeL_f": rl/n, "bertscore_f1": float(F1.mean())}
