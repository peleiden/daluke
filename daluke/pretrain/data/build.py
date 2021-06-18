from __future__ import annotations
import os
import shutil
import json
import multiprocessing as mp
import random
import re
from collections import defaultdict

from pelutils import log, TT
from tqdm import tqdm
from transformers import AutoTokenizer, XLMRobertaTokenizer, RobertaTokenizer
try:
    from wikipedia2vec.dump_db import DumpDB
    wikipedia2vec_available = True
except ImportError:
    wikipedia2vec_available = False


from daluke.pretrain.data import ICUSentenceTokenizer, load_entity_vocab, calculate_spans, ignore_title


class DatasetBuilder:

    tokenizer_language = "da"

    # Files saved by the build method
    metadata_file     = "metadata.json"
    data_file         = "data.jsonl"
    entity_vocab_file = "entity-vocab.json"

    def __init__(
        self,
        dump_db_file:        str,  # Location of file build by build-dump-db
        tokenizer_name:      str,  # Tokenizer to use, e.g. Maltehb/danish-bert-botxo for Danish BERT
        entity_vocab_file:   str,  # Build by build-entity-vocab
        out_dir:             str,  # Where to put finished dataset. All contents will be removed before saving dataset
        max_seq_length:      int,  # Maximum length of any sequence
        max_entities:        int,  # Only up to this many entities are included in each sequence
        max_entity_span:     int,  # Maximum number tokens an entity can span before sequence is discarded
        min_sentence_length: int,  # Minimum number of tokens a sentence must span to be included
        max_articles:        int | None,
    ):
        if not wikipedia2vec_available:
            raise RuntimeError("Pretrain data generation requires installation of the optional requirement `wikipedia2vec`")
        log("Reading dump database at %s" % dump_db_file)
        self.dump_db = DumpDB(dump_db_file)
        log("Building tokeninizer: %s" % tokenizer_name)
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        log("Building sentence tokenizer: %s" % self.tokenizer_language)
        self.sentence_tokenizer = ICUSentenceTokenizer(self.tokenizer_language)
        log("Loading entity vocab at %s" % entity_vocab_file)
        self.entity_vocab = load_entity_vocab(entity_vocab_file)
        # Make sure IDs on non-ignored entities are contiguous
        num = 0
        for entity, info in self.entity_vocab.items():
            if not ignore_title(entity):
                info["id"] = num
                num += 1
        log("Entity vocab has size %i" % num)

        self.out_dir             = out_dir
        self.max_seq_length      = max_seq_length
        self.max_entities        = max_entities
        self.max_entity_span     = max_entity_span
        self.min_sentence_length = min_sentence_length
        self.tokenizer_name      = tokenizer_name
        # Get maximum number of tokens in a sequence excluding [CLS] and [SEP]
        self.max_num_tokens = max_seq_length - 2
        self.max_articles = max_articles

        # Filter titles so only real articles are included
        self.target_titles = [title for title in self.dump_db.titles() if not ignore_title(title)]

    def _tokenize(self, text: str, paragraph_text: str, idx: int) -> list[str]:
        text = re.sub(r"\s+", " ", text).rstrip().lower()
        if not text:
            return []
        if isinstance(self.tokenizer, RobertaTokenizer):
            tokens = self.tokenizer.tokenize(
                text,
                add_prefix_space=idx == 0 or text.startswith(" ") or paragraph_text[idx-1] == " ",
            )
        else:
            tokens = self.tokenizer.tokenize(text)

        return tokens

    def build(self):
        log("Saving tokenizer config and word token config to %s" % self.out_dir)
        with open(path := os.path.join(self.out_dir, self.entity_vocab_file), "w", encoding="utf-8") as ev:
            log("Saving entity vocab to %s" % path)
            json.dump(self.entity_vocab, ev, indent=2)

        log.section("Processing pages")
        n_seqs, n_ents = 0, 0
        for title in log.tqdm(tqdm(self.target_titles[:self.max_articles])):
            log("Processing %s" % title)
            with TT.profile("Process page"):
                s, e = self._process_page(title)
                n_seqs += s
                n_ents += e

        # Save metadata
        with open(path := os.path.join(self.out_dir, self.metadata_file), "w") as f:
            log("Saving metadata to %s" % path)
            json.dump({
                "number-of-items":     n_seqs,
                "number-of-entities":  n_ents,
                "max-seq-length":      self.max_seq_length,
                "max-entities":        self.max_entities,
                "max-entity-span":     self.max_entity_span,
                "min-sentence-length": self.min_sentence_length,
                "base-model":          self.tokenizer_name,
                "tokenizer-class":     self.tokenizer.__class__.__name__,
                "language":            self.dump_db.language,
            }, f, indent=4)

        log.debug("Time distribution", TT)

    def _get_sentence_features(self, page_title: str) -> list[tuple[list[str], 3]]:

        sentences = list()

        # Process by paragraph
        for paragraph in self.dump_db.get_paragraphs(page_title):
            paragraph_links: list[tuple[str, int, int]] = list()
            paragraph_text = paragraph.text

            # Get paragraph links
            # These are representated by three-tuples consisting of their title, start and end string positions
            TT.profile("Get links")
            for link in paragraph.wiki_links:
                link_title: str = self.dump_db.resolve_redirect(link.title)
                # Remove category links
                if link_title.startswith("Kategori:") and link.text.lower().startswith("kategori:"):
                    paragraph_text = paragraph_text[:link.start]\
                        + " " * (link.end - link.start)\
                        + paragraph_text[link.end:]
                elif link_title in self.entity_vocab:
                    paragraph_links.append((link_title, link.start, link.end))
            TT.end_profile()

            # Process by sentence
            TT.profile("Sentences")
            sent_spans = self.sentence_tokenizer.span_tokenize(paragraph_text.rstrip())
            for sent_start, sent_end in sent_spans:
                current = sent_start
                sent_words = list()  # Tokens in the given sentence
                sent_links = list()  # Links in a given sentence in three-tuples: (id, start index, end index)

                too_large_tokens = False

                # Look for links that are within the tokenized sentence
                # If a link is found, the sentences are seperated across the link and tokenized
                for link_title, link_start, link_end in paragraph_links:
                    # Check if link is fully contained within sentence
                    if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                        continue

                    TT.profile("Tokenize")
                    text = paragraph_text[current:link_start]
                    sent_words += self._tokenize(text, paragraph_text, current)

                    link_text = paragraph_text[link_start:link_end]
                    link_words = self._tokenize(link_text, paragraph_text, link_start)
                    TT.end_profile()

                    sent_links.append((
                        self.entity_vocab[link_title]["id"],
                        len(sent_words),
                        len(sent_words) + len(link_words),
                    ))
                    if sent_links[-1][2] - sent_links[-1][1] > self.max_entity_span:
                        too_large_tokens = True
                        break
                    sent_words += link_words
                    current = link_end

                text = paragraph_text[current:sent_end]
                sent_words += self._tokenize(text, paragraph_text, current)

                if len(sent_words) >= self.min_sentence_length\
                    and len(sent_words) <= self.max_num_tokens\
                    and not too_large_tokens:
                    sentences.append((sent_words, sent_links))
            TT.end_profile()

        return sentences

    def _process_page(self, page_title: str) -> int:
        """
        Processes a Wikipedia article and save to self.data_file
        Returns number of sequences
        """

        sentences = self._get_sentence_features(page_title)

        # Construct features to be saved - word tokens, entities, and entity spans
        words = list()
        links: list[tuple[int, 3]] = list()
        n_seqs, n_ents = 0, 0
        TT.profile("Get features")
        for i, (sent_words, sent_links) in enumerate(sentences):
            links += [(id_, start + len(words), end + len(words)) for id_, start, end in sent_links]
            words += sent_words
            if i == len(sentences) - 1 or len(words) + len(sentences[i+1][0]) > self.max_num_tokens:
                if links:
                    n_seqs += 1
                    # Save features for this sequence
                    links = links[:self.max_entities]
                    n_ents += len(links)
                    word_ids = self.tokenizer.convert_tokens_to_ids(words)
                    with TT.profile("Word spans"):
                        word_spans = calculate_spans(words)
                    assert self.min_sentence_length <= len(word_ids) <= self.max_num_tokens
                    entity_ids = [id_ for id_, _, _ in links]
                    entity_spans = [(start, end) for _, start, end in links]
                    features = json.dumps({
                        "page_title":   page_title,
                        "word_ids":     word_ids,
                        "word_spans":   word_spans,
                        "entity_ids":   entity_ids,
                        "entity_spans": entity_spans,
                    })
                    with open(os.path.join(self.out_dir, self.data_file), "a") as df, TT.profile("Save features"):
                        df.write(features + "\n")
                words = list()
                links = list()
        TT.end_profile()

        return n_seqs, n_ents
