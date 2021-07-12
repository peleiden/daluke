from __future__ import annotations

from pelutils import log, Table
from pelutils.ds import no_grad
import numpy as np
import torch

from daluke.ner.model import span_probs_to_preds
from daluke.api.data import masked_example_from_str, ner_example_from_str, SingletonNERData
from daluke.api.automodels import AutoMLMDaLUKE, AutoNERDaLUKE

@no_grad
def predict_mlm(masked_text: str, entity_spans: list[tuple[int, int]], daluke: AutoMLMDaLUKE, k=5) -> tuple[str, Table]:
    """ Given a text containing [MASK] tokens, returns a text with [MASK] replaced by DaLUKE's best guesses """
    log.debug("Building example")
    # Make sure there are spaces between [MASK] tokens
    while "[MASK][MASK]" in masked_text:
        masked_text = masked_text.replace("[MASK][MASK]", "[MASK] [MASK]")
    example = masked_example_from_str(masked_text, entity_spans, daluke)
    log.debug("Forward passing example")
    probs = daluke.predict(example)
    top_k = np.argsort(probs.detach().cpu().numpy(), axis=1)[:, -k:]
    most_likely_ids = top_k[:, -1]
    tokens = daluke.tokenizer.convert_ids_to_tokens(most_likely_ids)
    for token in tokens:
        masked_text = masked_text.replace("[MASK]", token, 1)
    masked_text = masked_text.replace(" ##", "")

    t = Table()
    t.add_header(["[MASK] no.", "Top 1", "Top 2", "Top 3", "Top 4", "Top 5"])
    for i, top_ids in enumerate(top_k):
        row = [("%s - % 3.2f %%" % (daluke.tokenizer.convert_ids_to_tokens([top_id])[0], 100 * probs[i, top_id])) for top_id in reversed(top_ids)]
        t.add_row([i, *row], [True] + [False]*(k-1))

    return masked_text, t

@no_grad
def predict_ner(text: str, daluke: AutoNERDaLUKE) -> list[str]:
    """ Return list of entities corresponding """
    log.debug("Building example")
    example = ner_example_from_str(text, daluke)
    log.debug("Forward passing example")
    span_probs = daluke.predict(example)
    preds = span_probs_to_preds(span_probs, len(text.split()), SingletonNERData)
    return preds
