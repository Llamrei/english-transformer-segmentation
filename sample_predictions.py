import sys
import tensorflow as tf
import tensorflow_datasets as tfds

from pathlib import Path
from transformer_segmentation import Transformer
from transformer_segmentation import tokenizer_vocab_size
from transformer_segmentation import optimizer
from transformer_segmentation import NUM_LAYERS
from transformer_segmentation import D_MODEL
from transformer_segmentation import NUM_ATTENTION_HEADS
from transformer_segmentation import DFF
from transformer_segmentation import DROPOUT_RATE
from transformer_segmentation import NGRAM
from transformer_segmentation import MAX_CHARS
from transformer_segmentation import BATCH_SIZE
from transformer_segmentation import join_title_desc
from transformer_segmentation import unescape
from transformer_segmentation import strip_spaces_and_set_predictions

train, test = tfds.load('ag_news_subset', split="train"), tfds.load('ag_news_subset', split="test")
train_ds = train.shuffle(100).batch(BATCH_SIZE).map(join_title_desc).map(unescape).map(strip_spaces_and_set_predictions)

print(NGRAM, tokenizer_vocab_size(3))
encoder_tokenizer = tf.keras.layers.TextVectorization(
        standardize="lower", 
        split="character", 
        ngrams=(NGRAM,),
        max_tokens=tokenizer_vocab_size(3), # Want to be able to handle at least #, . and '
        output_sequence_length=MAX_CHARS-1, # Drop one as we need to account for the fact we predict if a space _follows_
        output_mode="int"
        )
def get_without_spaces(inputs, labels):
    return inputs[0]
inputs = train_ds.map(get_without_spaces)
encoder_tokenizer.adapt(inputs)

transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_attention_heads=NUM_ATTENTION_HEADS,
        dff=DFF,
        input_tokenizer=encoder_tokenizer,
        dropout_rate=DROPOUT_RATE
        )

transformer.compile(optimizer=optimizer, run_eagerly=True)
transformer.load_weights('./checkpoint/')

preds = transformer(
    (
        tf.constant([
            ['thecatsatonthematordidhe'],
            ['sowhatimstillarockstar']
        ]),
        [
            [0],
            [1]
        ],
    ),
    training=False
)

tf.print(preds, summarize=-1)