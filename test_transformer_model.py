import tensorflow as tf

import unittest

# Test input pipeline
from .transformer_segmentation import NGRAM
from .transformer_segmentation import MAX_CHARS
from .transformer_segmentation import strip_spaces_and_set_predictions
from .transformer_segmentation import sentence_accuracy
from .transformer_segmentation import Transformer


class TestInputPipeline(tf.test.TestCase):
    def test_mapping(self):
        prepend = "#"*(NGRAM-1)
        mappings = [
        (
            tf.constant([
                'the cat sat on the mat',
                'what if a word is not long'
            ])
            ,
            (
                (
                    tf.constant([
                        'thecatsatonthemat',
                        'whatifawordisnotlong'
                    ]),
                    tf.constant([
                        prepend+'the cat sat on the mat ',
                        prepend+'what if a word is not long '
                    ]),
                )
                ,
                tf.ragged.constant([
                    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                    [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1])             
            )
        ),
        ]
        for _input, expected_output in mappings:
            res = strip_spaces_and_set_predictions(_input, negative_control=False)
            with self.subTest("Labels correct"):
                self.assertAllClose(expected_output[1], res[1])
            with self.subTest("Encoder input correct"):            
                self.assertAllEqual(expected_output[0][0], res[0][0])
            with self.subTest("Decoder input correct"):        
                self.assertAllEqual(expected_output[0][1], res[0][1])

    def test_negative_control(self):
        prepend = "#"*(NGRAM-1)
        mappings = [
        (
            tf.constant([
                'the cat sat on the mat',
                'what if a word is not long'
            ])
            ,
            (
                (
                    tf.constant([
                        'thecatsatonthemat',
                        'whatifawordisnotlong'
                    ]),
                    tf.constant([
                        prepend+'the cat sat on the mat ',
                        prepend+'what if a word is not long '
                    ]),
                )
                ,
                tf.ragged.constant([
                    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                    [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1])             
            )
        ),
        ]
        for _input, expected_output in mappings:
            res = strip_spaces_and_set_predictions(_input, negative_control=True)
            mask = tf.cast(expected_output[1] != 0, tf.float32)
            spaces = tf.cast((res[1] - 1), mask.dtype)*mask
            av_spaces = tf.reduce_mean(tf.reduce_sum(spaces, axis=-1)/tf.reduce_sum(mask, axis=-1))
            with self.subTest("Labels correct"):
                self.assertBetween(av_spaces, 0.4, 0.6) # TODO: figure out a proper 99% CI
            with self.subTest("Encoder input correct"):
                self.assertAllEqual(expected_output[0][0], res[0][0])
            with self.subTest("Decoder input correct"):
                self.assertAllEqual(expected_output[0][1], res[0][1])


class TestMetrics(tf.test.TestCase):
    def test_sentence_accuracy(self):
        mappings = [
            (
                (
                    tf.ragged.constant([
                        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                        [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                    ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                    tf.ragged.constant([
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    ], dtype=tf.float32).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                )
                ,
                tf.constant(0.5)
            ),
            (
                (
                    tf.ragged.constant([
                        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                        [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                    ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                    tf.ragged.constant([
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
                    ], dtype=tf.float32).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                )
                ,
                tf.constant(1.0)
            ),
            (
                (
                    tf.ragged.constant([
                        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                        [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                    ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                    tf.ragged.constant([
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
                    ], dtype=tf.float32).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                )
                ,
                tf.constant(0.0)
            ),
        ]

        for _input, expected_output in mappings:
            labels, preds = _input
            res = sentence_accuracy(labels, preds)
            self.assertAllEqual(expected_output, res)

    def test_sentence_accuracy_metric(self):
        mappings = [
            (
                (
                    tf.ragged.constant([
                        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                        [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                    ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                    tf.ragged.constant([
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    ], dtype=tf.float32).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                )
                ,
                tf.constant(0.5)
            ),
            (
                (
                    tf.ragged.constant([
                        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                        [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                    ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                    tf.ragged.constant([
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
                    ], dtype=tf.float32).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                )
                ,
                tf.constant(0.75) # Averaged with previous, (0.5 + 1) / 2
            ),
            (
                (
                    tf.ragged.constant([
                        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                        [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                    ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                    tf.ragged.constant([
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
                    ], dtype=tf.float32).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                )
                ,
                tf.constant(0.5) # Averaged with previous (0.5 + 1 + 0)/3
            ),
        ]
        train_sentence_accuracy = tf.keras.metrics.Mean(name='train_token_accuracy')
        for _input, expected_output in mappings:
            labels, preds = _input
            res = sentence_accuracy(labels, preds)
            train_sentence_accuracy(res)
            self.assertAllEqual(expected_output, train_sentence_accuracy.result())
    

class TestInference(tf.test.TestCase):
    @unittest.skip("Dont want to always run this")
    def setUp(self):
        tf.random.set_seed(150797)
        self.encoder_tokenizer = tf.keras.layers.TextVectorization(
            standardize="lower", 
            split="character", 
            ngrams=(NGRAM,),
            max_tokens=5,
            output_sequence_length=MAX_CHARS-1,
            output_mode="int",
            vocabulary=["###", "##t", "cat"]
            )
        self.decoder_tokenizer = tf.keras.layers.TextVectorization(
            standardize="lower", 
            split="character", 
            ngrams=(NGRAM,),
            max_tokens=5,
            output_sequence_length=MAX_CHARS-1,
            output_mode="int",
            vocabulary=["###", "##t", "cat"]
            )
        self.inputs = tf.constant([
                    'thecatsatonthemat',
                    'whatifawordisnotlong'
                ])
        self.outputs = tf.constant([
                    "#"*(NGRAM-1)+'the cat sat on the mat',
                    "#"*(NGRAM-1)+'what if a word is not long'
                ])
    
    @unittest.skip("Dont want to always run this")
    def testForwardPass(self):
        with self.session(use_gpu=False):
            tf.random.set_seed(150797)
            model = Transformer(
                num_layers=1,
                d_model=32,
                num_attention_heads=1,
                dff=64,
                input_tokenizer=self.encoder_tokenizer,
                target_tokenizer=self.decoder_tokenizer,
                dropout_rate=0.1
            )
            res = model([self.inputs, self.outputs], training=False)
            tf.print(res, summarize = -1)
            self.assertEqual(1,1)