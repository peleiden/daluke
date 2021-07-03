from __future__ import annotations

from pelutils import log, Table
from pelutils.ds import no_grad
from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn.functional as F

from daluke.api.fetch_model import fetch_model, Models
from daluke.api.data import masked_example_from_str, ner_example_from_str, SingletonNERData, example_from_str
from daluke.ner.model import span_probs_to_preds
from daluke.pretrain.data.masking import BatchedExamples


def predict_mlm(masked_text: str, entity_spans: list[tuple[int, 2]]) -> tuple[str, Table]:
    """ Given a text containing [MASK] tokens, returns a text with [MASK] replaced by DaLUKE's best guesses """
    log.debug("Loading model")
    model, metadata, entity_vocab = fetch_model(Models.DaLUKE)
    tokenizer = AutoTokenizer.from_pretrained(metadata["base-model"])
    log.debug("Building example")
    # Make sure there are spaces between [MASK] tokens
    while "[MASK][MASK]" in masked_text:
        masked_text = masked_text.replace("[MASK][MASK]", "[MASK] [MASK]")
    example = masked_example_from_str(masked_text, entity_spans, entity_vocab, metadata)
    log.debug("Forward passing example")
    word_scores, __ = model(example)
    probs = F.softmax(word_scores, dim=1)
    top_k = np.argsort(probs.detach().cpu().numpy(), axis=1)[:, -5:]
    most_likely_ids = top_k[:, -1]
    tokens = tokenizer.convert_ids_to_tokens(most_likely_ids)
    for token in tokens:
        masked_text = masked_text.replace("[MASK]", token, 1)
    masked_text = masked_text.replace(" ##", "")

    t = Table()
    t.add_header(["[MASK] no.", "Top 1", "Top 2", "Top 3", "Top 4", "Top 5"])
    for i, top_ids in enumerate(top_k):
        row = [("%s - % 3.2f %%" % (tokenizer.convert_ids_to_tokens([top_id])[0], 100 * probs[i, top_id])) for top_id in reversed(top_ids)]
        t.add_row([i, *row], [1, 0, 0, 0, 0, 0])

    return masked_text, t

def predict_ner(text: str) -> list[str]:
    """ Return list of entities corresponding """
    log.debug("Loading model")
    model, metadata, __ = fetch_model(Models.DaLUKE_NER)
    log.debug("Building example")
    example = ner_example_from_str(text, metadata)
    log.debug("Forward passing example")
    scores = model(example)
    probs = F.softmax(scores, dim=2)
    # We save probability distribution, for every possible span in the example
    span_probs = dict()
    for i, spans in enumerate(example.entities.fullword_spans):
        span_probs.update({
            span: probs[i, j].detach().cpu().numpy() for j, span in enumerate(spans) if span
        })
    preds = span_probs_to_preds(span_probs, len(text.split()), SingletonNERData)
    return preds

@no_grad
def get_contextualized_representations(batch: BatchedExamples) -> tuple[torch.Tensor, torch.Tensor]:
    """ Get contextualied word and entity representations. Gradients are not tracked """
    model, __, __ = fetch_model(Models.DaLUKE)
    return super(type(model), model).forward(batch)
