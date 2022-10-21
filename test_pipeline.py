import tensorflow as tf

# Test input pipeline
from .transformer_segmentation import NGRAM
from .transformer_segmentation import MAX_CHARS
from .transformer_segmentation import strip_spaces_and_set_predictions


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
                        prepend+'thecatsatonthemat ',
                        prepend+'whatifawordisnotlong '
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
        for input, expected_output in mappings:
            res = strip_spaces_and_set_predictions(input, negative_control=False)
            self.assertAllClose(expected_output[1], res[1])
            self.assertAllEqual(expected_output[0][0], res[0][0])
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
                        prepend+'thecatsatonthemat ',
                        prepend+'whatifawordisnotlong '
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
        for input, expected_output in mappings:
            res = strip_spaces_and_set_predictions(input, negative_control=True)
            mask = tf.cast(expected_output[1] != 0, tf.float32)
            spaces = tf.cast((res[1] - 1), mask.dtype)*mask
            av_spaces = tf.reduce_mean(tf.reduce_sum(spaces, axis=-1)/tf.reduce_sum(mask, axis=-1))
            self.assertBetween(av_spaces, 0.4, 0.6) # TODO: figure out a proper 99% CI
            self.assertAllEqual(expected_output[0][0], res[0][0])
            self.assertAllEqual(expected_output[0][1], res[0][1])


