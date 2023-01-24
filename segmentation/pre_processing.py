# TODO: Re-architect this parametrisation through top-of-script globals

import tensorflow as tf

JOINING_PUNC = r"([-'`])"
SPLITTING_PUNC = r'([!"#$%&()\*\+,\./:;<=>?@\[\\\]^_{|}~])'
NGRAM = 3 if not DEBUG else 1
MAX_CHARS = 500 if not DEBUG else 50
BATCH_SIZE = 8 if not DEBUG else 2


def strip_spaces_and_set_predictions(text, negative_control=NEGATIVE_CONTROL, positive_control=POSITIVE_CONTROL):
    x = tf.strings.lower(text)
    x = tf.strings.substr(x, 0, MAX_CHARS)
    # We want to sometimes replace punctuation with a space : "bad.sentence" -> "bad. sentence"
    # and other time to replace it with nothing: "don't" -> "dont"
    x = tf.strings.regex_replace(x, JOINING_PUNC, "")
    x = tf.strings.regex_replace(x, SPLITTING_PUNC, r"\1 ")   # \1 inserts captured splitting punctuation, raw so python doesnt magic it
    if positive_control:
        x = tf.strings.regex_replace(x, r"\s", "") # remove all whitespace
        logger.info("Running a positive control experiment")
        x = tf.strings.regex_replace(x, r"(e)", r"\1 ") # add whitespace back in after every letter e
    
    x = tf.strings.split(x)

    no_whitespace_list = tf.strings.strip(x) # Remove any excess whitespace, e.g. tabs, double spaces etc.
    word_lengths = tf.strings.length(no_whitespace_list)
    def mark_spaces(row_of_word_lengths):
        
        def all_zeros_except_last_letter(word_length):
            a = tf.zeros((word_length), dtype=tf.float32)
            a = tf.concat([a[:-1],[1.]], axis=-1)
            return a
        
        binary_chars = tf.map_fn(
            fn=all_zeros_except_last_letter,
            elems=row_of_word_lengths,
            fn_output_signature = tf.RaggedTensorSpec([None], dtype=tf.float32)
        )
        binary_chars = binary_chars.merge_dims(0, 1)
        return binary_chars
        
    y = tf.map_fn(
        fn=mark_spaces,
        elems=word_lengths,
        fn_output_signature = tf.RaggedTensorSpec([None])
    )
    
    y_start = tf.broadcast_to(1., (tf.shape(y)[0],1) )  # Essentially start token
    y = tf.concat([y_start, y], axis=-1)

    x = tf.strings.reduce_join(no_whitespace_list, separator="", axis=-1)
    x_start = tf.broadcast_to("#"*NGRAM, (tf.shape(x)[0],) )  # Essentially start token
    x = tf.strings.reduce_join([x_start, x], axis=0)

    y = y + 1 # Add 1 to tell difference between labels and missing
    y = y.to_tensor(default_value=0, shape=[None, MAX_CHARS])
    # Roll so we predict for _future_ spaces not spaces in current token (if we knew space in current token we wouldnt have anything to solve)
    labels = tf.roll(y, shift=-1, axis=-1)
    labels = labels[:,:-1]

    # Negative controls
    # Permute
    # labels = tf.random.experimental.stateless_shuffle(labels, seed=[1507, 1997]) # shuffles batch

    # Truly random
    if negative_control:
        logger.info("Running a negative control experiment")
        present = labels != 0
        labels = tf.random.stateless_binomial(shape=tf.shape(labels), seed=[1507, 1997], counts=1, probs=0.5) + 1
        labels = tf.cast(labels, y.dtype)
        labels = tf.cast(present, labels.dtype)*labels
    return (x, y[:,:-1]), labels # We don't need to try to predict last token as we know there is a space