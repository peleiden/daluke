from __future__ import annotations
from typing import TextIO
import os
import pickle
import random

import numpy as np
import torch
import ujson
from pelutils import log, TT
from pelutils.ds import unique
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaTokenizer
from daluke.data import get_special_ids
try:
    from wikipedia2vec.dump_db import DumpDB
    wikipedia2vec_available = True
except ImportError:
    wikipedia2vec_available = False

from daluke.pretrain.data import ICUSentenceTokenizer,\
    load_entity_vocab, calculate_spans, ignore_title


class DatasetBuilder:

    tokenizer_language = "da"

    # Files saved by the build method
    metadata_file     = "metadata.json"
    entity_vocab_file = "entity-vocab.json"
    data_file         = "data.jsonl"
    token_map_file    = "token-map.npy"

    def __init__(
        self,
        dump_db_file:        str,    # Location of file build by build-dump-db
        tokenizer_name:      str,    # Tokenizer to use, e.g. Maltehb/danish-bert-botxo for Danish BERT
        entity_vocab_file:   str,    # Build by build-entity-vocab
        out_dir:             str,    # Where to put finished dataset. All contents will be removed before saving dataset
        validation_prob:     float,  # Chance of each finished document to be marked as part of validation set
        max_entities:        int,    # Only up to this many entities are included in each sequence
        max_entity_span:     int,    # Maximum number tokens an entity can span before sequence is discarded
        min_sentence_length: int,    # Minimum number of tokens a sentence must span to be included
        max_articles:        int | None,
        max_vocab_size:      int,
    ):
        if not wikipedia2vec_available:
            raise ModuleNotFoundError("Pretrain data generation requires installation of the optional requirement `wikipedia2vec`")
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
        for entity_info in self.entity_vocab.values():
            entity_info["id"] = num
            num += 1
        log("Entity vocab has size %i" % num)

        self.out_dir             = out_dir
        self.data_file           = os.path.join(self.out_dir, self.data_file)
        self.token_map_file      = os.path.join(self.out_dir, self.token_map_file)
        self.max_seq_length      = self.tokenizer.model_max_length
        self.validation_prob     = validation_prob
        self.max_entities        = max_entities
        self.max_entity_span     = max_entity_span
        self.min_sentence_length = min_sentence_length
        # Get maximum number of tokens in a sequence excluding start and end tokens
        self.max_num_tokens      = self.max_seq_length - 2
        self.max_articles        = max_articles
        self.vocab_size          = self.tokenizer.vocab_size if max_vocab_size == -1 else min(max_vocab_size, max_vocab_size)

        # Filter titles so only real articles are included
        self.target_titles = list(self.dump_db.titles())

        # Remove old datafile if it exists
        if os.path.isfile(self.data_file):
            log.debug("Removing old datafile '%s'" % self.data_file)
            os.remove(self.data_file)

        self.examples = list()

    def _tokenize(self, text: str, paragraph_text: str, idx: int) -> list[str]:
        if not text:
            return list()
        try:
            if isinstance(self.tokenizer, RobertaTokenizer):
                tokens = self.tokenizer.tokenize(
                    text,
                    add_prefix_space=idx == 0 or text.startswith(" ") or paragraph_text[idx-1] == " ",
                )
            else:
                tokens = self.tokenizer.tokenize(text)
        except KeyboardInterrupt:
            # Make sure program can be keyboard interrupted despite needing to catch BaseException
            raise
        except BaseException as e:
            # Catch an exception caused by rust panicking in the tokenizer
            log.warning("Failed to tokenize text with exception '%s'\nText: '%s'" % (e, text))
            return list()

        return tokens

    def build(self):
        log("Saving tokenizer config and word token config to '%s'" % self.out_dir)
        with open(path := os.path.join(self.out_dir, self.entity_vocab_file), "w", encoding="utf-8") as ev:
            log("Saving entity vocab to '%s'" % path)
            ujson.dump(self.entity_vocab, ev, indent=2)

        log.section("Processing %i pages" % len(self.target_titles[:self.max_articles]))
        n_seqs, n_ents, n_word_toks, n_words = 0, 0, 0, 0
        for title in log.tqdm(tqdm(self.target_titles[:self.max_articles])):
            log("Processing %s" % title)
            with TT.profile("Process page"):
                s, e, nt, nw = self._process_page(title)
                n_seqs += s
                n_ents += e
                n_word_toks += nt
                n_words += nw

        log("Shuffling data")
        random.shuffle(self.examples)
        n_vals = int(self.validation_prob*len(self.examples))
        for i in range(n_vals):
            self.examples[i]["is_validation"] = True

        # Save metadata
        metadata = {
            "number-of-items":       n_seqs,
            "number-of-word-tokens": n_word_toks,
            "number-of-words":       n_words,
            "number-of-entities":    n_ents,
            "number-of-val-items":   n_vals,
            "max-seq-length":        self.max_seq_length,
            "max-entities":          self.max_entities,
            "max-entity-span":       self.max_entity_span,
            "min-sentence-length":   self.min_sentence_length,
            "base-model":            self.tokenizer_name,
            "tokenizer-class":       self.tokenizer.__class__.__name__,
            "language":              self.dump_db.language,
            "reduced-vocab":         self.vocab_size < self.tokenizer.vocab_size,
            "vocab-size":            self.vocab_size,
        }

        if self.vocab_size < self.tokenizer.vocab_size:
            log.section("Reducing token number")
            with TT.profile("Reduce token vocab"):
                token_map, metadata["vocab-size"] = self._reduce_tokens()
            with TT.profile("Rewrite dataset with new tokens"):
                self._update_tokens(token_map)

        with open(path := os.path.join(self.out_dir, self.metadata_file), "w") as f:
            log.section("Saving metadata to %s" % path)
            ujson.dump(metadata, f, indent=4)
        with open(self.data_file, "w") as f, TT.profile("Save data"):
            log("Saving data to '%s'" % self.data_file)
            for example in self.examples:
                f.write(ujson.dumps(example)+"\n")

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
                # Remove links to articles that are not included
                if ignore_title(link_title):
                    paragraph_text = paragraph_text[:link.start]\
                        + " " * (link.end - link.start)\
                        + paragraph_text[link.end:]
                elif link_title in self.entity_vocab:
                    paragraph_links.append((link_title, link.start, link.end))
            paragraph_links = list(reversed(paragraph_links))
            TT.end_profile()

            # Process by sentence
            TT.profile("Sentences")
            if paragraph_links:
                link_title, link_start, link_end = paragraph_links.pop()
            else:
                link_title, link_start, link_end = "", -1, -1

            sent_spans = self.sentence_tokenizer.span_tokenize(paragraph_text.rstrip())
            for sent_start, sent_end in sent_spans:
                current = sent_start
                sent_words = list()  # Tokens in the given sentence
                sent_links = list()  # Links in a given sentence in three-tuples: (id, start index, end index)
                too_large_tokens = False

                while link_start < sent_start:
                    try:
                        link_title, link_start, link_end = paragraph_links.pop()
                    except IndexError:
                        break

                while sent_start <= link_start and link_end <= sent_end:
                    # Look for links that are within the tokenized sentence
                    # If a link is found, the sentences are seperated across the link and tokenized
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

                    try:
                        link_title, link_start, link_end = paragraph_links.pop()
                    except IndexError:
                        break

                text = paragraph_text[current:sent_end]
                sent_words += self._tokenize(text, paragraph_text, current)

                if len(sent_words) >= self.min_sentence_length\
                    and len(sent_words) <= self.max_num_tokens\
                    and not too_large_tokens:
                    sentences.append((sent_words, sent_links))
            TT.end_profile()

        return sentences

    def _process_page(self, page_title: str) -> tuple[int, int, int, int]:
        """ Processes a Wikipedia article and save to self.data_file
        Returns number of sequences """

        sentences = self._get_sentence_features(page_title)

        # Construct features to be saved - word tokens, entities, and entity spans
        words = list()
        links: list[tuple[int, 3]] = list()
        n_seqs, n_ents, n_word_toks, n_words = 0, 0, 0, 0
        TT.profile("Get features")
        for i, (sent_words, sent_links) in enumerate(sentences):
            links += [(id_, start + len(words), end + len(words)) for id_, start, end in sent_links]
            words += sent_words
            if i == len(sentences) - 1 or len(words) + len(sentences[i+1][0]) > self.max_num_tokens:
                n_seqs += 1
                # Save features for this sequence
                links = links[:self.max_entities]
                n_ents += len(links)
                word_ids = self.tokenizer.convert_tokens_to_ids(words)
                with TT.profile("Word spans"):
                    word_spans = calculate_spans(words, self.tokenizer)
                assert self.min_sentence_length <= len(word_ids) <= self.max_num_tokens
                entity_ids = [id_ for id_, _, _ in links]
                entity_spans = [(start+1, end+1) for _, start, end in links]
                # Whether to mark doc. as part of validation set
                n_word_toks += len(word_ids)
                n_words += len(word_spans)
                self.examples.append({
                    "word_ids":      [self.tokenizer.cls_token_id, *word_ids, self.tokenizer.sep_token_id],
                    "word_spans":    word_spans,
                    "entity_ids":    entity_ids,
                    "entity_spans":  entity_spans,
                    "is_validation": False,
                })
                words = list()
                links = list()
        TT.end_profile()

        return n_seqs, n_ents, n_word_toks, n_words

    def _reduce_tokens(self) -> tuple[np.ndarray, int]:
        token_counts = np.zeros(self.tokenizer.vocab_size, dtype=np.int32)

        log("Counting tokens in dataset")
        for example in tqdm(self.examples):
            word_ids = np.array(example["word_ids"])
            word_ids, counts = unique(word_ids, return_counts=True)
            token_counts[word_ids] += counts

        log("%i of %i tokens in the vocab are used" % ((token_counts>0).sum(), self.tokenizer.vocab_size))
        *ids, unk_id = get_special_ids(self.tokenizer)
        unk_count = token_counts[unk_id]
        token_counts[[*ids, unk_id]] = -1
        sort_idx = np.argsort(token_counts)[::-1]
        keep_idx = sort_idx[:self.vocab_size]
        keep = np.zeros_like(token_counts, dtype=bool)
        keep[keep_idx] = True
        keep[[*ids, unk_id]] = True  # Always keep special tokens
        token_map = np.arange(self.tokenizer.vocab_size)
        token_map[~keep] = unk_id
        for i, j in enumerate(np.where(keep)[0]):
            token_map[j] = i
        log(
            "Reduced token vocabulary to %i tokens" % keep.sum(),
            "%.6f %% of word tokens in the dataset are now %s" % (
                100 * (unk_count + 1 + token_counts[~keep].sum()) / (unk_count + 1 + token_counts.sum()),
                self.tokenizer.unk_token,
            ),
        )
        np.save(self.token_map_file, token_map)
        log("Saved token map to '%s'" % self.token_map_file)

        return token_map, int(keep.sum())

    def _update_tokens(self, token_map: np.ndarray):
        log("Updating dataset with kept tokens")
        for example in tqdm(self.examples):
            example["word_ids"] = token_map[example["word_ids"]].tolist()
