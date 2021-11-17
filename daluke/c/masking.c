#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>


double randd() {
    /* Samples a random number in [0, 1] */
    return (double)rand() / (double)RAND_MAX;
}

int randint(int min, int max) {
    /* Samples a random int in [min, max[ */
    return rand() % (max-min) + min;
}

int mask_words(
    int mask_id,
    int rand_min_id,
    int rand_max_id,  // Inclusive
    double mask_prob,
    double unmask_prob,
    double randword_prob,
    int batchsize,
    int max_seq_length,
    int max_num_words,
    int* ids,
    unsigned char* mask,
    int* spans,
    int* num_words,
    int* labels
) {
    /* Masks words
    ids: batch size x max_seq_length
    mask: batch size x max_seq_length
    spans: batch size x max_num_words x 2
    num_words: number of word spans in each example
    Returns number of labels */
    srand(time(NULL));
    int num_labels = 0;

    size_t i, j, k;
    for (i = 0; i < batchsize; i ++) {
        int* seq_ids = ids + i * max_seq_length;
        int* seq_spans = spans + i * max_num_words * 2;
        unsigned char* seq_mask = mask + i * max_seq_length;

        for (j = 0; j < num_words[i]; j ++) {
            if (randd() < mask_prob) {
                // printf("Word %zu\n", j);
                int word_start = seq_spans[j*2];
                int word_end = seq_spans[j*2+1];
                int num_word_tokens = word_end - word_start;
                double p = randd();
                if (p > 1 - randword_prob) {
                    // Do random token replacement
                    for (k = word_start; k < word_end; k ++) {
                        seq_ids[k] = randint(rand_min_id, rand_max_id);
                    }
                } else if (p > unmask_prob) {
                    // No unmasking, so do masking
                    memcpy(labels, seq_ids+word_start, num_word_tokens*sizeof(*ids));
                    wmemset(seq_ids+word_start, mask_id, num_word_tokens);
                    memset(seq_mask+word_start, 1, num_word_tokens);
                    labels += num_word_tokens;
                    num_labels += num_word_tokens;
                }
            }
        }
    }

    return num_labels;
}
