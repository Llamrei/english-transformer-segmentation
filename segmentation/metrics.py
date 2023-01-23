import tensorflow as tf

class SparsePrecision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            y_true,
            tf.argmax(y_pred, axis=-1)
            )

class SparseRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            y_true,
            tf.argmax(y_pred, axis=-1)
            )

class SparseAccuracyWithIgnore(tf.keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name="sparse_categorical_accuracy", dtype=None, ignore_token=None):
        self.ignore_token = ignore_token
        super().__init__(name, dtype)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if not tf.is_tensor(y_true):
            y_true = tf.constant(y_true)
        
        if not tf.is_tensor(y_pred):
            y_pred = tf.Variable(y_pred)
        
        if self.ignore_token is not None:
            mask = tf.not_equal(y_true, tf.cast(self.ignore_token, y_true.dtype))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        return super().update_state(y_true, y_pred, sample_weight)