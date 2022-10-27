#!/usr/bin/env python
# coding: utf-8

# Next step: Convert the model to be subclass of Model
# 1) Rewrite train step, using custom metrics and returning in a dict, thus not needing to use custom Metrics
# 2) Rewrite loss as tensorflow Loss function
# 3) Rewrite metrics that occur only on X iterations: inspired by https://stackoverflow.com/questions/56826495/how-to-make-keras-compute-a-certain-metric-on-validation-data-only
# 4) Write test time/call logic
# 5) Do we need to return attention weights?
# Then: Run with Tensorboard profiler plugin
# Also: think about how to create dataset for Kevin's version
# Also: rewrite a token accuracy metric that corresponds to BIES tagging protocol
# Think why NoTwoSpaces is so slow

from datetime import datetime
import logging
import logging.handlers as handlers
import pathlib
import time
import sys
import string
import html

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf # v2.10
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

# TODO:
# Lot of scope for optimisation of data types as we dont need more than int8 or something to store 1s/0s
# Also could improve through use of gather to make the where code more streamlined
# Also can improve speed of metrics through converting into tf.function - but need to remove python mutable objects
# Can also probably speed up metrics through use of tensorarrays instead of ragged tensors.
# Proper Hyper param opt and experiment tracking
# More precisely accurate loss using backend.binary_crossentropy

TODAY = datetime.today().strftime('%Y-%m-%d %H-%M')
TENSORBOARD_DIR = './tensorboard/'
CHECKPOINT_DIR = './checkpoint/'


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stderr = logging.StreamHandler()
stderr.setLevel(logging.INFO)
stderr.setFormatter(formatter)
file_output = handlers.RotatingFileHandler(f'{TODAY}.log', maxBytes=10000, backupCount=1)
file_output.setLevel(logging.DEBUG)
file_output.setFormatter(formatter)
logger.addHandler(file_output)

DEBUG = True
NEGATIVE_CONTROL = False

logger.info(f"Eager: {tf.executing_eagerly()}")

# Running params
if __name__ == "__main__":
    RUN_AS_SCRIPT = True
    GPU_FROM = int(sys.argv[1])
    GPU_TO = int(sys.argv[2])
else:
    RUN_AS_SCRIPT = False
    GPU_FROM = 0
    GPU_TO = 0

# Data definition params 
JOINING_PUNC = r"([-'`])"
SPLITTING_PUNC = r'([!"#$%&()\*\+,\./:;<=>?@\[\\\]^_{|}~])'
NGRAM = 1
MAX_CHARS = 1000 if not DEBUG else 100
BATCH_SIZE = 32 if not DEBUG else 2

# Metric params
CLASS_THRESHOLD = 0.5

# Model params
NUM_LAYERS = 4
D_MODEL = 256
DFF = 1028
NUM_ATTENTION_HEADS = 2
DROPOUT_RATE = 0.3

# Training params
EPOCHS = 30 if not DEBUG else 300
STEPS_PER_EPOCH = 100 if not DEBUG else 2
VALIDATION_STEPS = 50 if not DEBUG else 2


visible_devices = tf.config.get_visible_devices('GPU')
logger.info(f"Num GPUs visible:{len(visible_devices)}")
tf.config.set_visible_devices(visible_devices[GPU_FROM:GPU_TO],'GPU')

visible_devices = tf.config.get_visible_devices('GPU')
logger.info(f"Num GPUs to be used: {len(visible_devices)}")

# strategy = tf.distribute.MirroredStrategy(devices=visible_devices) # Look at https://www.tensorflow.org/tutorials/distribute/custom_training for more guide on how to do this


# # Data preprocessing
# Create dataset that converts texts into rolling trigram windows
# reminder: Assuming 26 english characters -> 26^3 = 18k combos to learn
# at some point we might want to start keeping capitalisation (could be important for distinguishing acronyms)
# How to handle punctuation and numbers?

# For real deal probably want the 'lm1b' dataset
logger.info("Loading data")
train, test = tfds.load('ag_news_subset', split="train"), tfds.load('ag_news_subset', split="test")

logger.info("Preprocessing data")
punc_mapping = {ord(x):x for x in string.punctuation}
entity_mapping = {f" ?{k}": v for k, v in html.entities.html5.items() if k.endswith(";") and v in string.punctuation}
punc_mapping = {f' ?&?#?{k};': v for k, v in punc_mapping.items()}

def unescape(text):
    for match, replace in punc_mapping.items():
        text = tf.strings.regex_replace(text, match, replace)
    for match, replace in entity_mapping.items():#
        text = tf.strings.regex_replace(text, match, replace)
    return text

def join_title_desc(text):
    return text['title'] + ' ' + text['description']

def strip_spaces_and_set_predictions(text, negative_control=NEGATIVE_CONTROL):
    x = tf.strings.lower(text)
    x = tf.strings.substr(x, 0, MAX_CHARS)
    # We want to sometimes replace punctuation with a space : "bad.sentence" -> "bad. sentence"
    # and other time to replace it with nothing: "don't" -> "dont"
    x = tf.strings.regex_replace(x, JOINING_PUNC, "")
    x = tf.strings.regex_replace(x, SPLITTING_PUNC, r"\1 ")   # \1 inserts captured splitting punctuation, raw so python doesnt magic it
    x = tf.strings.split(x)
    no_whitespace_list = tf.strings.strip(x) # Remove any excess whitespace, e.g. tabs, double spaces etc.
    x = tf.strings.reduce_join(no_whitespace_list, separator="", axis=-1)
    y = tf.strings.reduce_join(no_whitespace_list, separator=" ", axis=-1)
    
    y_padding = tf.broadcast_to("#"*(NGRAM-1), (tf.shape(x)[0],) )  # Essentially my start token
    y = tf.strings.reduce_join([y_padding, y], axis=0)
    
    final_boundary = tf.broadcast_to(" ", (tf.shape(x)[0],) )
    y = tf.strings.reduce_join([y, final_boundary], axis=0)
    
    chars = tf.strings.unicode_split(y, "UTF-8")
    char_ngrams = tf.strings.ngrams(chars, ngram_width=NGRAM, separator="")
    labels = tf.strings.regex_full_match(char_ngrams, ".* ")
    labels = tf.cast(labels, tf.int64)+1 # Add 1 to be able to tell difference between padding and prediction
    labels = labels.to_tensor(default_value=0, shape=[None, MAX_CHARS])
    # Roll so we predict for _future_ spaces not spaces in current token (if we knew space in current token we wouldnt have anything to solve)
    labels = tf.roll(labels, shift=-1, axis=-1)
    labels = labels[:,:-1]

    # Negative controls
    # Permute
    # labels = tf.random.experimental.stateless_shuffle(labels, seed=[1507, 1997]) # shuffles batch

    # Truly random
    if negative_control:
        logger.info("Running a negative control experiment")
        present = labels != 0
        labels = tf.random.stateless_binomial(shape=tf.shape(labels), seed=[1507, 1997], counts=1, probs=0.5) + 1
        labels = tf.cast(labels, tf.int64)
        labels = tf.cast(present, labels.dtype)*labels
    return (x, y), labels

train_ds = train.shuffle(100).batch(BATCH_SIZE).map(join_title_desc).map(unescape).map(strip_spaces_and_set_predictions)
test_ds = test.batch(BATCH_SIZE).map(join_title_desc).map(unescape).map(strip_spaces_and_set_predictions)
# train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
# test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

if DEBUG:
    logger.debug("Generating pipepline label stats")
    def label_stats(inputs, labels):
        mask = tf.cast(labels != 0, tf.float32)
        spaces = tf.cast(labels == 2, tf.float32)
        def batchwise_mean(tensor, mask):
            return tf.reduce_mean(tf.reduce_sum(tensor, axis=-1)/tf.reduce_sum(mask, axis=-1))
        
        return (
            batchwise_mean(mask, tf.ones_like(mask)),
            batchwise_mean(spaces, mask)
        )

    avg_char_usage = tf.constant(0.)
    avg_no_spaces = tf.constant(0.)
    i = tf.constant(0)
    for char_usage, no_spaces in train_ds.take(100).map(label_stats):
        avg_char_usage += char_usage
        avg_no_spaces += no_spaces
        i += 1
    i = tf.cast(i, tf.float32)
    avg_char_usage /= i
    avg_no_spaces /= i
    logger.debug(f"Avg char_usage {avg_char_usage}")
    logger.debug(f"Avg no spaces {avg_no_spaces}")


logger.info("Training tokenizers")
# # Defining tokenizers
# Want to be able to encode every n-gram of lowercase letters and however many prescribed punctuation characters
def tokenizer_vocab_size(num_punc):
    if DEBUG:
        return 100
    return (len(string.ascii_lowercase)+num_punc)**NGRAM
    
encoder_tokenizer = tf.keras.layers.TextVectorization(
    standardize="lower", 
    split="character", 
    ngrams=(NGRAM,),
    max_tokens=tokenizer_vocab_size(3), # Want to be able to handle at least #, . and '
    output_sequence_length=MAX_CHARS-1, # Drop one as we need to account for the fact we predict if a space _follows_
    output_mode="int"
    )
decoder_tokenizer = tf.keras.layers.TextVectorization(
    standardize="lower", 
    split="character", 
    ngrams=(NGRAM,),
    max_tokens=tokenizer_vocab_size(4), # As above but also with a space
    output_sequence_length=MAX_CHARS-1,
    output_mode="int"
    )
def get_without_spaces(inputs, labels):
    return inputs[0]
def get_with_spaces(inputs, labels):
    return inputs[1]


if RUN_AS_SCRIPT:
    if DEBUG:
        inputs = train_ds.take(200).map(get_without_spaces)
        outputs = train_ds.take(200).map(get_with_spaces)
        encoder_tokenizer.adapt(inputs)
        decoder_tokenizer.adapt(outputs)
    else:
        inputs = train_ds.map(get_without_spaces)
        outputs = train_ds.map(get_with_spaces)
        encoder_tokenizer.adapt(inputs)
        decoder_tokenizer.adapt(outputs)

# # Metrics and losses
loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    reduction=tf.keras.losses.Reduction.NONE # Need no reduction so we can mask out irrelevant predictions
)

def get_real(real_raw):
    """
    Converts a batch of {0,1,2} into a batch of {0,1} by mapping 1 to 0 and 2 to 1.

    Also returns a batched boolean mask corresponding to locations of original 1s and 2s.
    """
    # Function to convert the 0 mask and 1/2 category tokens into a mask 
    # and 0/1 tokens (suitable for binary cross-entropy)
    # Mask will have bool dtype
    mask = tf.math.logical_not(tf.math.equal(real_raw, 0))
    real = real_raw - tf.cast(mask, real_raw.dtype)
    return real, mask

def used_all_characters_mask(real_raw, pred):
    """
    Takes a label batch of {0,1,2} indicating not present, letter, space 
    and a prediction batch of [0,1] indicating letter and space.

    Returns batch mask of pred that represents using all the letters in the label.
    """
    real, _ = get_real(real_raw) # batch, max_chars
    real_letters = 1 - real
    real_letter_count = tf.math.cumsum(real_letters, axis=-1)
    letters_in_phrase = real_letter_count[..., -1] # batch, 1
    
    if tf.shape(pred)[-1] == 1:
        pred = tf.squeeze(pred, axis=-1) # pred now batch, max_chars
    boundaries = tf.cast(pred > CLASS_THRESHOLD, real.dtype) # batch, max_chars
    pred_letters = 1 - boundaries
    pred_letter_count = tf.math.cumsum(pred_letters, axis=-1) # batch, max_chars
    return pred_letter_count <= letters_in_phrase[..., tf.newaxis] # batch max_chars

def loss_function(real, pred, mask):
    """
    Takes a batch of {0,1} labels (e.g. need to preprocess labels), [0,1] predictions and a boolean mask for computing loss.

    Also returns a distribution of the loss across the batch.
    """
    if tf.shape(pred)[-1] == 1:
        pred = tf.squeeze(pred, axis=-1)
    pred *= tf.cast(mask, dtype=pred.dtype)
    loss_object.reduction = tf.losses.Reduction.NONE # Make loss not reduce so we can sample dist of loss
    loss_ = loss_object(real, pred)
    # Dont divide by tf.cast(tf.reduce_sum(mask), dtype=loss_.dtype) 
    # as we hope that the masked (0 + eps) output matching the real 0 is good enough to not fit to masked data
    return tf.reduce_mean(loss_), loss_

def token_accuracy(real_raw, pred):
    """
    Takes a batch of labels {0,1,2} and batch of predictions [0,1] and returns per token accuracy.

    We can mask out either following only tokens that are in label or tokens in prediction
    that represent usage of all of the letters. We choose whichever is longer.
    """
    real, real_mask = get_real(real_raw)
    predicted_mask = used_all_characters_mask(real_raw, pred)
    mask = tf.cast(predicted_mask | real_mask, real.dtype)
    # tf.print(real_raw, real, pred, mask, summarize=-1)
    if tf.shape(pred)[-1] == 1:
        pred = tf.squeeze(pred, axis=-1)
    pred *= tf.cast(mask, pred.dtype)
    accuracies = tf.equal(real, tf.cast( pred > CLASS_THRESHOLD, real.dtype))
    accuracies = tf.cast(accuracies, dtype=mask.dtype)
    accuracies *= mask
    return tf.reduce_mean(tf.reduce_sum(accuracies, axis=-1)/tf.reduce_sum(mask, axis=-1))

def sentence_accuracy(labels, pred):
    """
    Takes a batch of labels {0,1,2} and batch of predictions [0,1] and returns whole sentence accuracy across the batch.

    We can mask out either following only tokens that are in label or tokens in prediction
    that represent usage of all of the letters. We choose whichever is longer.

    Dont need to do fancy masking because if the whole sentence is correct it will just all line up anyway
    on the real mask.
    """
    real, real_mask = get_real(labels)
    mask = tf.cast(real_mask, real.dtype)
    if tf.shape(pred)[-1] == 1:
        pred = tf.squeeze(pred, axis=-1)
    accuracies = tf.equal(real, tf.cast(pred > CLASS_THRESHOLD, real.dtype))
    accuracies = tf.cast(accuracies, dtype=mask.dtype)
    accuracies *= mask
    accuracies += 1 - mask # A bit worried about the double accuracy of this but w/e
    sentence_accuracies = tf.reduce_prod(accuracies, axis=-1)
    return tf.reduce_mean(sentence_accuracies)

def row_parse_real_word_boundaries(real_row):
    # Given (seq_length,) we want to return a tensor of shape
    # (no_words, 2) where first index chooses a word and 2nd index chooses which
    # numbered letter starts and ends the word
    real, _ = get_real(real_row)
    no_words = tf.reduce_sum(real, axis=-1) # We don't need to find the first boundary
    real = tf.roll(real, shift=1, axis=-1)
    real_letters = 1 - real
    real_letter_count = tf.cast(tf.math.cumsum(real_letters), real.dtype)
    words = []
    start_index = tf.cast(0, tf.int64)
    for end_index in tf.where(real):
        end_index = end_index[0]
        words.extend([real_letter_count[start_index],real_letter_count[end_index]])
        start_index = end_index + 1
    res = tf.RaggedTensor.from_uniform_row_length(words, 2)
    return res

def row_indices_of_wrapping_boundaries(pred_row, real_word_boundaries):
    # If real_word_boundaries is (no_words, 2) and pred is (seq_length, 1)
    # Then this should return (no_words, 3) second index idenitfying
    # left gap, mid word gap, right gap
    if tf.shape(pred_row)[-1] == 1:
        pred_row = tf.squeeze(pred_row, axis=-1)
    pred_spaces = tf.cast(pred_row > CLASS_THRESHOLD, tf.int64)
    seq_length = tf.cast(tf.shape(pred_row)[0], dtype=pred_spaces.dtype)
    pred_indices = tf.range(seq_length, dtype=pred_spaces.dtype)
    # Mask out last prediction before rolling
    pred_spaces *= tf.cast(pred_indices < seq_length - 1, dtype=pred_spaces.dtype)
    pred_spaces = tf.roll(pred_spaces, shift=1, axis=-1)
    pred_letters = 1 - pred_spaces 
    pred_letter_count = tf.math.cumsum(pred_letters)
    pred_letter_count -= pred_spaces*tf.cast(tf.shape(pred_row)[0]*2, pred_spaces.dtype)
    wrapping_boundaries = []
    for word_index in tf.range(tf.shape(real_word_boundaries)[0]):
        word = tf.cast(real_word_boundaries[word_index,...], pred_spaces.dtype)
        start_count, end_count = tf.unstack(word)
        word_mask = (pred_letter_count >= start_count) & (pred_letter_count <= end_count)
        letter_indices = tf.where(word_mask)
        if tf.shape(letter_indices)[0] >= tf.cast((end_count - start_count)+1, tf.int32):
            start_index, end_index = letter_indices[0][0], letter_indices[-1][0]
        else:
            # Can't find the word in this prediction - means we are definitely fuarked
            # We pretend it is a total failure by saying the string is the whole string minus two either side so falls
            # into cat 6 error
            start_index, end_index = tf.constant(2, dtype=pred_spaces.dtype), tf.cast(tf.shape(pred_spaces)[0]-2, dtype=pred_spaces.dtype)
        left_mask = tf.cast(pred_indices < start_index, pred_spaces.dtype)
        right_mask = tf.cast(pred_indices > end_index, pred_spaces.dtype)
        if tf.shape(tf.where(tf.cast(left_mask & pred_spaces, tf.int64)))[0] > 0:
            left_boundary = tf.where(tf.cast(left_mask & pred_spaces, tf.int64))[-1][0]
        else:
            left_boundary = tf.constant(-1, dtype=pred_spaces.dtype)
        if tf.shape(tf.where(tf.cast(right_mask & pred_spaces, tf.int64)))[0] > 0:
            right_boundary = tf.where(tf.cast(right_mask & pred_spaces, tf.int64))[0][0]
        else:
            right_boundary = tf.cast(tf.shape(pred_spaces)[0]+1, dtype=pred_spaces.dtype)
        left_gap = start_index - left_boundary
        letter_gap = end_index - start_index
        letter_delta = letter_gap - (end_count - start_count)
        right_gap = right_boundary - end_index
        wrapping_boundaries.extend([left_gap, letter_delta, right_gap])
    return tf.RaggedTensor.from_uniform_row_length(wrapping_boundaries, 3)    

def invalid_gaps(word_info):
    return (word_info[0] < 1) | (word_info[2] < 1)

def no_error(word_info):
    return (word_info[0] == 1) & (word_info[1] == 0) & (word_info[2] == 1)

def one_side_no_tile(word_info):
    left_gap = (word_info[0] > 1) & (word_info[1] == 0) & (word_info[2] == 1)
    right_gap = (word_info[0] == 1) & (word_info[1] == 0) & (word_info[2] > 1)
    return left_gap | right_gap

def gap_no_tile(word_info):
    return (word_info[0] > 1) & (word_info[1] == 0) & (word_info[2] > 1)

def perfect_tile(word_info):
    return (word_info[0] == 1) & (word_info[1] > 0) & (word_info[2] == 1)

def one_side_tile(word_info):
    left_gap = (word_info[0] > 1) & (word_info[1] > 0) & (word_info[2] == 1)
    right_gap = (word_info[0] == 1) & (word_info[1] > 0) & (word_info[2] > 1)
    return left_gap | right_gap

def worst_case(word_info):
    return (word_info[0] > 1) & (word_info[1] > 0) & (word_info[2] > 1)

def classify_errors(real, pred):
    # If real is (batch, seq_length) and pred is (batch, seq_length, 1)
    # Then this should return (batch, 6) 
    # Where second index identifies error types for words in this predicted sentence
    
    # As of TF 2.10 tf.map_fn is just a niceity around tf.while_loop
    batch_size = tf.shape(pred)[0]
    res = tf.TensorArray(tf.float64, size=batch_size, dynamic_size=False)
    for sample_index in tf.range(batch_size):
        error_types = np.zeros(shape=(6,))
        word_locs = row_parse_real_word_boundaries(real[sample_index,...])
        match_info = row_indices_of_wrapping_boundaries(pred[sample_index,...], word_locs)
        num_words = tf.shape(match_info)[0]
        for word_index in tf.range(num_words):
            word_info = match_info[word_index,:]
            if invalid_gaps(word_info):
                raise ValueError(f"Cannot have a left/right gap < 0: {word_info}")
            elif no_error(word_info):
                error_types[0] += 1
            elif one_side_no_tile(word_info):
                error_types[1] += 1
            elif gap_no_tile(word_info):
                error_types[2] += 1
            elif perfect_tile(word_info):
                error_types[3] += 1
            elif one_side_tile(word_info):
                error_types[4] += 1
            elif worst_case(word_info):
                error_types[5] += 1
        error_types = error_types
        res = res.write(sample_index, tf.constant(error_types, dtype=res.dtype))
    return res.stack()

class VotingExpertsMetric(tf.keras.metrics.Metric):
    def __init__(self, name="voting_experts", **kwargs):
        super().__init__(name=name, **kwargs)
        self.error_types = self.add_weight(name="error_types", shape=(6,))

    def update_state(self, real, pred):
        errors = classify_errors(real, pred)
        self.error_types.assign_add(tf.cast(tf.reduce_sum(errors, axis=0), self.error_types.dtype)) # Add across batch
        
    def result(self):
        normalized = self.error_types/tf.reduce_sum(self.error_types)
        return {
            'good': normalized[0],
            'one-sided-gap': normalized[1],
            'two-sided-gap': normalized[2],
            'tiled-no-gap': normalized[3],
            'tiled-one-gap': normalized[4],
            'totally-wrong': normalized[5]
        }

# # Embedding defn
def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth/2)
    angle_rates = 1 / (10000**depths)         # (1, depth/2)
    angle_rads = positions * angle_rates      # (pos, depth/2)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)], # (pos, depth)
      axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=MAX_CHARS, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


# # Custom output layer to enforce no two spaces
class NoTwoSpaces(tf.keras.layers.Layer):    
    def __init__(self, classification_threshold):
        super().__init__()
        self.threshold = classification_threshold
    
    @tf.function
    def remove_consecutive_ones(self, row):
        # We now work on a single row, if two consecutive 1s appear we
        # want to replace with 0 -> only 1*1 =/= 0
        return tf.scan(lambda a, t: t*(1-tf.cast(a>self.threshold, a.dtype)*tf.cast(t>self.threshold, t.dtype)), row)
    
    @tf.function
    def call(self, x): # Do I need to support masking?
        return tf.map_fn(
            lambda row: self.remove_consecutive_ones(row),
            x
        )

# # Transformer building blocks

def point_wise_feed_forward_network(
    d_model, # Input/output dimensionality.
    dff # Inner-layer dimensionality.
    ):

    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # Shape `(batch_size, seq_len, dff)`.
      tf.keras.layers.Dense(d_model)  # Shape `(batch_size, seq_len, d_model)`.
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               dropout_rate=0.1
               ):
        super().__init__()


        # Multi-head self-attention.
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model, # Size of each attention head for query Q and key K.
            dropout=dropout_rate,
            )
        # Point-wise feed-forward network.
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Layer normalization.
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):

        # A boolean mask.
        if mask is not None:
            mask1 = mask[:, :, None]
            mask2 = mask[:, None, :]
            attention_mask = mask1 & mask2
        else:
            attention_mask = None

        # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_output = self.mha(
            query=x,  # Query Q tensor.
            value=x,  # Value V tensor.
            key=x,  # Key K tensor.
            attention_mask=attention_mask, # A boolean mask that prevents attention to certain positions.
            training=training, # A boolean indicating whether the layer should behave in training mode.
            )

        # Multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(x + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out1)  # Shape `(batch_size, input_seq_len, d_model)`
        ffn_output = self.dropout1(ffn_output, training=training)
        # Point-wise feed-forward network output after layer normalization and a residual skip connection.
        out2 = self.layernorm2(out1 + ffn_output)  # Shape `(batch_size, input_seq_len, d_model)`.

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
               *,
               num_layers,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               tokenizer, # Input (Portuguese) vocabulary size.
               dropout_rate=0.1
               ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Tokenization
        self.tokenizer = tokenizer

        # Embeddings + Positional encoding
        self.pos_embedding = PositionalEmbedding(tokenizer.vocabulary_size(), d_model)

        # Encoder layers.
        self.enc_layers = [
            EncoderLayer(
              d_model=d_model,
              num_attention_heads=num_attention_heads,
              dff=dff,
              dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        # Dropout.
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    # Masking.
    def compute_mask(self, x, previous_mask=None):
        x = self.tokenizer(x)
        return self.pos_embedding.compute_mask(x, previous_mask)

    def call(self, x, training):
        # Sum up embeddings and positional encoding.
        x = self.tokenizer(x)
        mask = self.pos_embedding.compute_mask(x)
        x = self.pos_embedding(x)  # Shape `(batch_size, input_seq_len, d_model)`.
        # Add dropout.
        x = self.dropout(x, training=training)

        # N encoder layers.
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # Shape `(batch_size, input_seq_len, d_model)`.

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
               *,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               dropout_rate=0.1
               ):
        super().__init__()

        # Masked multi-head self-attention.
        self.mha_masked = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model, # Size of each attention head for query Q and key K.
            dropout=dropout_rate
        )
        # Multi-head cross-attention.
        self.mha_cross = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model, # Size of each attention head for query Q and key K.
            dropout=dropout_rate
        )

        # Point-wise feed-forward network.
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Layer normalization.
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, mask, enc_output, enc_mask, training):
        # The encoder output shape is `(batch_size, input_seq_len, d_model)`.

        # A boolean mask.
        self_attention_mask = None
        if mask is not None:
            mask1 = mask[:, :, None]
            mask2 = mask[:, None, :]
            self_attention_mask = mask1 & mask2

        # Masked multi-head self-attention output (`tf.keras.layers.MultiHeadAttention`).
        attn_masked, attn_weights_masked = self.mha_masked(
            query=x,
            value=x,
            key=x,
            attention_mask=self_attention_mask,  # A boolean mask that prevents attention to certain positions.
            use_causal_mask=True,  # A boolean to indicate whether to apply a causal mask to prevent tokens from attending to future tokens.
            return_attention_scores=True,  # Shape `(batch_size, target_seq_len, d_model)`.
            training=training  # A boolean indicating whether the layer should behave in training mode.
            )

        # Masked multi-head self-attention output after layer normalization and a residual/skip connection.
        out1 = self.layernorm1(attn_masked + x)

        # A boolean mask.
        attention_mask = None
        if mask is not None and enc_mask is not None:
            mask1 = mask[:, :, None]
            mask2 = enc_mask[:, None, :]
            attention_mask = mask1 & mask2

        # Multi-head cross-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_cross, attn_weights_cross = self.mha_cross(
            query=out1,
            value=enc_output,
            key=enc_output,
            attention_mask=attention_mask,  # A boolean mask that prevents attention to certain positions.
            return_attention_scores=True,  # Shape `(batch_size, target_seq_len, d_model)`.
            training=training  # A boolean indicating whether the layer should behave in training mode.
        )

        # Multi-head cross-attention output after layer normalization and a residual/skip connection.
        out2 = self.layernorm2(attn_cross + out1)  # (batch_size, target_seq_len, d_model)

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out2)  # Shape `(batch_size, target_seq_len, d_model)`.
        ffn_output = self.dropout1(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # Shape `(batch_size, target_seq_len, d_model)`.

        return out3

class Decoder(tf.keras.layers.Layer):
    def __init__(self,
               *,
               num_layers,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               tokenizer,
               dropout_rate=0.1
               ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.tokenizer = tokenizer
        self.pos_embedding = PositionalEmbedding(tokenizer.vocabulary_size(), d_model)

        self.dec_layers = [
            DecoderLayer(
              d_model=d_model,
              num_attention_heads=num_attention_heads,
              dff=dff,
              dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    # Masking.
#     def compute_mask(self, x, previous_mask=None):
#         x = self.tokenizer(x)
#         return self.pos_embedding.compute_mask(x, previous_mask)
    
    def call(self, dec_input, enc_output, enc_mask, training):
        x = self.tokenizer(dec_input)
        mask = self.pos_embedding.compute_mask(x)
        x = self.pos_embedding(x)  # Shape: `(batch_size, target_seq_len, d_model)`.

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, mask, enc_output, enc_mask, training)

        # The shape of x is `(batch_size, target_seq_len, d_model)`.
        return x


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_loss_low = tf.keras.metrics.Mean(name='train_loss_low')
train_loss_med = tf.keras.metrics.Mean(name='train_loss_med')
train_loss_high = tf.keras.metrics.Mean(name='train_loss_high')

train_token_accuracy = tf.keras.metrics.Mean(name='train_token_accuracy')
train_sentence_accuracy = tf.keras.metrics.Mean(name='train_token_accuracy')
train_AUC = tf.keras.metrics.AUC(name="train_AUC", from_logits=False)
train_Voting_Experts = VotingExpertsMetric()

train_step_signature = [
    (
        (
            tf.TensorSpec(shape=(None, ), dtype=tf.string),
            tf.TensorSpec(shape=(None, ), dtype=tf.string)
        ),
        tf.TensorSpec(shape=(None, MAX_CHARS-1), dtype=tf.int64),
    )
]
train_writer = tf.summary.create_file_writer(TENSORBOARD_DIR + "train/")

# # Actual model

class Transformer(tf.keras.Model):
    def __init__(self,
               *,
               num_layers, # Number of decoder layers.
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               input_tokenizer,
               target_tokenizer,
               dropout_rate=0.1,
               classification_threshold=0.5
               ):
        super().__init__()
        # The encoder.
        self.encoder = Encoder(
          num_layers=num_layers,
          d_model=d_model,
          num_attention_heads=num_attention_heads,
          dff=dff,
          tokenizer=input_tokenizer,
          dropout_rate=dropout_rate
          )

        # The decoder.
        self.decoder = Decoder(
          num_layers=num_layers,
          d_model=d_model,
          num_attention_heads=num_attention_heads,
          dff=dff,
          tokenizer=target_tokenizer,
          dropout_rate=dropout_rate
          )

        # The final linear layer.
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")        
        self.final_layer = NoTwoSpaces(classification_threshold)

    def call(self, inputs, training=False):
        # Keras models prefer if you pass all your inputs in the first argument.
        to_enc, to_dec = inputs

        # The encoder output.
        enc_output = self.encoder(to_enc, training)  # `(batch_size, inp_seq_len, d_model)`
        enc_mask = self.encoder.compute_mask(to_enc)

        if training:
            # The decoder output.
            dec_output = self.decoder(
                to_dec, enc_output, enc_mask, training)  # `(batch_size, tar_seq_len, d_model)`

            # The final linear layer output.
            final_output = self.dense(dec_output)  # Shape `(batch_size, tar_seq_len, 1)`.
            # no_two_spaces = self.final_layer(final_output)

            # Return the final output and the attention weights.
            return final_output
        else:
            # Essentially while we still have characters to place spaces 
            # in we want to predict the decoder output
            # Starting with just the token '###'
            # Start simple by doing each element in the batch individually
            batch_size = tf.shape(to_enc)[0]
            output = tf.TensorArray(dtype=tf.float32, size=batch_size)
            for sample_inx in tf.range(batch_size):
                sample = to_enc[sample_inx:sample_inx+1, ...] # (1, string)
                chars_seen = 0
                to_dec = tf.strings.substr(sample, 0, NGRAM + chars_seen) # (1, string)
                
                
                preds = tf.TensorArray(dtype=tf.float32, size=MAX_CHARS-1)
                step = 0
                N = tf.strings.length(sample)[0] # Scalar
                while chars_seen < N and step < (MAX_CHARS - 1):
                    dec_output = self.decoder(
                        to_dec, enc_output[sample_inx:sample_inx+1,...], enc_mask[sample_inx:sample_inx+1,...], training
                    ) # (1, tar_seq_len, d_model)
                    spaces = self.dense(dec_output) # (1, tar_seq_len, 1)
                    space = spaces[0, step, 0]
                    preds = preds.write(step, space)
                    if  space > CLASS_THRESHOLD:
                        to_dec = tf.strings.join([to_dec, ' '])
                    else:
                        to_dec = tf.strings.join([
                            to_dec, 
                            tf.strings.substr(sample, NGRAM + chars_seen, 1)
                        ])
                        chars_seen += 1
                    step += 1
                output = output.write(sample_inx, preds.stack())
            return output.stack()[..., tf.newaxis]
                
    @tf.function(input_signature=train_step_signature)
    def train_step(self, data):
        inputs, labels = data
        real, real_mask = get_real(labels)

        with tf.GradientTape() as tape:
            preds = self(inputs, training = True)
            predicted_mask = used_all_characters_mask(labels, preds)
            loss, loss_dist = loss_function(real, preds, predicted_mask | real_mask)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        lq, med, uq = tfp.stats.percentile(loss_dist, 10), tfp.stats.percentile(loss_dist, 50), tfp.stats.percentile(loss_dist, 90)
        train_loss_low(med-lq)
        train_loss_med(med)
        train_loss_high(uq-med)
        train_loss(loss)
        train_token_accuracy(token_accuracy(labels, preds))
        train_sentence_accuracy(sentence_accuracy(labels, preds))
        metrics = {
            'loss': train_loss.result(),
            'median loss': train_loss_med.result(),
            'bottom decile delta': train_loss_low.result(),
            'top decile delta': train_loss_high.result(),
            'token accuracy': train_token_accuracy.result(),
            'sentence accuracy': train_sentence_accuracy.result(),
        }
        with train_writer.as_default(step=self._train_counter):
            for k, v in metrics.items():
                tf.summary.scalar(k, v)
        return metrics
    
    @tf.function(input_signature=train_step_signature)
    def test_step(self, data):
        inputs, labels = data
        real, real_mask = get_real(labels)

        preds = self(inputs, training = False)
        predicted_mask = used_all_characters_mask(labels, preds)
        loss, loss_dist = loss_function(real, preds, predicted_mask | real_mask)
        
        lq, med, uq = tfp.stats.percentile(loss_dist, 10), tfp.stats.percentile(loss_dist, 50), tfp.stats.percentile(loss_dist, 90)
        train_loss_low(med-lq)
        train_loss_med(med)
        train_loss_high(uq-med)
        train_loss(loss)
        train_token_accuracy(token_accuracy(labels, preds))
        train_sentence_accuracy(sentence_accuracy(labels, preds))
        metrics = {
            'loss': train_loss.result(),
            'median loss': train_loss_med.result(),
            'bottom decile delta': train_loss_low.result(),
            'top decile delta': train_loss_high.result(),
            'token accuracy': train_token_accuracy.result(),
            'sentence accuracy': train_sentence_accuracy.result(),
        }
        with train_writer.as_default(step=self._train_counter):
            for k, v in metrics.items():
                tf.summary.scalar(k, v)
        return metrics
    
    @property
    def metrics(self):
        return [train_loss, train_loss_med, train_loss_low, train_loss_high, train_token_accuracy, train_sentence_accuracy]

    
    


# # Model creation + training
# ## Hyperparams and model instantiation
logger.info("Creating transformer")
transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_attention_heads=NUM_ATTENTION_HEADS,
    dff=DFF,
    input_tokenizer=encoder_tokenizer,
    target_tokenizer=decoder_tokenizer,
    dropout_rate=DROPOUT_RATE
    )

# ## Training details
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=200):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_DIR,
    monitor='token accuracy',
    mode='max',
    save_freq=10,
    save_best_only=True,
    save_weights_only=True)

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = TENSORBOARD_DIR,
                                                #  histogram_freq = 50,
                                                 profile_batch = '75,125',
                                                 update_freq='batch',
                                                 write_steps_per_second=True,
                                                 embeddings_freq=50,
                                                )


# If a checkpoint exists, restore the latest checkpoint.
if not DEBUG and len(list(pathlib.Path(CHECKPOINT_DIR).glob("*"))) > 0:
    transformer.load_weights(CHECKPOINT_DIR)
    logger.info('Latest checkpoint restored!!')

callbacks=[model_checkpoint_callback,]
if DEBUG:
    callbacks.append(tboard_callback)

if RUN_AS_SCRIPT:
    logger.info("Compiling model")
    transformer.compile(optimizer=optimizer, run_eagerly=DEBUG)
    logger.info("Fitting model")
    transformer.fit(
        train_ds,
        callbacks=callbacks,
        validation_data=test_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        epochs=EPOCHS
        )