import tensorflow as tf

import os
import numpy as np
import random
import unittest

# Test input pipeline
from .transformer_segmentation import NGRAM
from .transformer_segmentation import MAX_CHARS
from .transformer_segmentation import strip_spaces_and_set_predictions
from .transformer_segmentation import sentence_accuracy
from .transformer_segmentation import precision_and_recall
from .transformer_segmentation import Transformer

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

set_global_determinism(150797)

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

    def test_precision_recall(self):
        mappings = [
            (
                (
                    tf.ragged.constant([
                        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                        [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                    ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                    tf.ragged.constant([
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    ], dtype=tf.float32).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                )
                ,
                (
                    # All predicted positives are real positives in both cases - precision fine
                    tf.constant([
                        6/6,
                        2/2,
                    ]),
                    # First row has all real positives recalled, but second only has 2 out of 7
                    tf.constant([
                        6/6,
                        2/7,
                    ]),
                ),
                "Good precision, bad recall"
                
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
                (
                    # All predicted positives are real positives in both cases
                    tf.constant([
                        6/6,
                        7/7,
                    ]),
                    # All real values recalled
                    tf.constant([
                        6/6,
                        7/7,
                    ]),
                ),
                "Good overall"
            ),
            (
                (
                    tf.ragged.constant([
                        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                        [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                    ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                    tf.ragged.constant([
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    ], dtype=tf.float32).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                )
                ,
                (
                    # Both cases have multiple predicted value that is not a true positive
                    tf.constant([
                        1/4,
                        1/3
                    ]),
                    tf.constant([
                        1/6,
                        1/7
                    ]),
                ),
                "Both poor"
            ),
            (
                (
                    tf.ragged.constant([
                        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2],
                        [1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2]
                    ], dtype=tf.int64).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                    tf.ragged.constant([
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                    ], dtype=tf.float32).to_tensor(default_value=0, shape=[None, MAX_CHARS-1]) ,
                )
                ,
                (
                    # Both cases have multiple predicted value that is not a true positive
                    tf.constant([
                        1/4,
                        1/3
                    ]),
                    tf.constant([
                        1/6,
                        1/7
                    ]),
                ),
                "Both poor - preds starting with space"
            ),
        ]

        for _input, expected_output, msg in mappings:
            labels, preds = _input
            pres, reca = precision_and_recall(labels, preds)
            with self.subTest("Precision: "+msg):
                self.assertAllClose(expected_output[0], pres, msg="Precision")
            with self.subTest("Recall: "+msg):
                self.assertAllClose(expected_output[1], reca)

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
    
    # @unittest.skip("Dont want to always run this")
    def testForwardPass(self):
        with self.session(use_gpu=False):
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
            expected = tf.ragged.constant([
                [
                    [0.113609634],
                    [0.247582987],
                    [0.295219243],
                    [0.212690353],
                    [0.110841691],
                    [0.0747087225],
                    [0.075634],
                    [0.0711343586],
                    [0.0726035684],
                    [0.0548635833],
                    [0.0308177676],
                    [0.0194270574],
                    [0.0226549748],
                    [0.0392052568],
                    [0.0681078658],
                    [0.0712439865],
                    [0.0420827083],
                ],
                [
                    [0.114118628],
                    [0.247462049],
                    [0.294691652],
                    [0.212689832],
                    [0.111605726],
                    [0.0749583393],
                    [0.0758525655],
                    [0.0720633715],
                    [0.0740739554],
                    [0.0557910949],
                    [0.0308968555],
                    [0.0191844888],
                    [0.0224053487],
                    [0.0397413075],
                    [0.0703945383],
                    [0.0736368522],
                    [0.0432050824],
                    [0.0272174198],
                    [0.029272031],
                    [0.0617184639],
                ]], dtype=tf.float32).to_tensor(default_value=0, shape=[None, MAX_CHARS-1, 1])
            self.assertAllClose(res, expected)