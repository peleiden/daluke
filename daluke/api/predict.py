from __future__ import annotations

from typing import Union, List

from pelutils import Table
from pelutils.ds import no_grad
import numpy as np

from daluke.ner.model import span_probs_to_preds
from daluke.api.data import masked_example_from_str, ner_examples_from_str, SingletonNERData, DEFAULT_NER_BATCHSIZE
from daluke.api.automodels import AutoMLMDaLUKE, AutoNERDaLUKE

@no_grad
def predict_mlm(masked_text: str, entity_spans: list[tuple[int, int]], daluke: AutoMLMDaLUKE, k=5) -> tuple[str, Table]:
    """ Given a text containing [MASK] tokens, returns a text with [MASK] replaced by DaLUKE's best guesses """
    # Make sure there are spaces between [MASK] tokens
    while "[MASK][MASK]" in masked_text:
        masked_text = masked_text.replace("[MASK][MASK]", "[MASK] [MASK]")
    example = masked_example_from_str(masked_text, entity_spans, daluke)
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

T = Union[str, List[str]]
@no_grad
def predict_ner(text: T, daluke: AutoNERDaLUKE, batch_size: int=DEFAULT_NER_BATCHSIZE) -> list[T]:
    """
    Given a text (or a list of texts), NER classes are predicted.
    """
    single_example = isinstance(text, str)
    texts: list[str] = [text] if single_example else text
    prepared_data = ner_examples_from_str(texts, daluke, batch_size=batch_size)
    if single_example:
        return span_probs_to_preds(daluke.predict(prepared_data[0]), len(text.split()), SingletonNERData)

    # The other case: We have to glue multiple spans from multiple examples together in the multi-document situation
    span_probs = list(dict() for _ in range(len(texts)))
    for batch in prepared_data:
        batch_probs = daluke.predict(batch, multiple_documents=True)
        # It must be handled that a single text can be divided into multiple batches
        for text_num, example_probs in batch_probs.items():
            span_probs[text_num].update(example_probs)
    return [span_probs_to_preds(p, len(t.split()), SingletonNERData) for p, t in zip(span_probs, texts)]
