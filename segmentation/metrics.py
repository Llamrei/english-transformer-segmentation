import tensorflow as tf

# TODO: write tests, e.g.

# m = SparsePrecision(0, class_id=2)

# # Simulating batch dimension 2
# m.update_state(
#     [
#         [1, 2, 1, 2, 0],
#         [1, 2, 1, 2, 0],
#     ],
#     [
#         [[1, 1, 5], [4,5,10], [-5,10,0], [0,10,-10], [100,2,1]], # Essentially [2, 2, 1, 1, 0]
#         [[1, 1, 5], [4,5,10], [-5,10,0], [0,10,-10], [100,2,1]], 
#     ]
# )
# m.result().numpy()

class SparsePrecision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        class_dim = tf.shape(y_pred)[-1]
        return super().update_state(
            tf.one_hot(tf.cast(y_true, tf.int32), depth=class_dim),
            tf.one_hot(tf.math.argmax(y_pred, axis=-1), depth=class_dim)
            )

class SparseRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        class_dim = tf.shape(y_pred)[-1]
        return super().update_state(
            tf.one_hot(tf.cast(y_true, tf.int32), depth=class_dim),
            tf.one_hot(tf.math.argmax(y_pred, axis=-1), depth=class_dim)
            )

class SparseAccuracyWithIgnore(tf.keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name="sparse_categorical_accuracy", dtype=None, ignore_token=None):
        self.ignore_token = ignore_token
        super().__init__(name, dtype)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.ignore_token:
            mask = tf.not_equal(y_true, tf.cast(self.ignore_token, y_true.dtype))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        return super().update_state(y_true, y_pred, sample_weight)

class SparseF1(tf.keras.metrics.Metric):
    def __init__(self, name=None, class_id=None, **kwargs):
        super().__init__(name, **kwargs)
        self.precision = SparsePrecision(class_id=class_id)
        self.recall = SparseRecall(class_id=class_id)
        self.f1 = self.add_weight('f1', initializer="zeros", dtype="float32")
    
    def update_state(self, *args, **kwargs):
        self.precision.update_state(*args, **kwargs)
        self.recall.update_state(*args, **kwargs)
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        
    def result(self):
        return 2*(self.precision.result()*self.recall.result()/(self.precision.result()+self.recall.result()))