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

def special_divide(a,b,default_val=1.):
  """
    Divide such that a/b = 0 if b == 0 and a != 0; `default_val` if a==b==0 and a/b otherwise.

    Allows for usage in precision and recall calcs when there is no positive cases in denom.
  """
  ans = tf.math.divide_no_nan(a,b)
  ans = tf.where(tf.logical_and(b==0, a!=0), 0., ans)
  ans = tf.where(tf.logical_and(b==0, a==0), default_val, ans)
  return ans

class SparsePrecision(tf.keras.metrics.Metric):
  def __init__(self, name="precision", class_id=1, batch_size=64, **kwargs):
    super().__init__(name=name, **kwargs)
    self.precision = self.add_weight(name="precision", initializer="zeros", dtype=tf.float32)
    self.i = self.add_weight(name="i", initializer="zeros")
    self.class_id = class_id

  def update_state(self, y_true, y_pred, sample_weight=None):
    """
      y_true assumed (Batch, Seq)
      y_pred assumed (Batch, Seq, Classes) where Classes is a distribution on classes

      Assumes fixed and constant batch size
      Convert distribution of y_pred to prediction via argmax, thus can also handle logits.
    """
    preds = tf.argmax(y_pred, axis=-1) # (B, S)
    class_preds = tf.where(preds==self.class_id, 1., 0.) # (B, S)
    class_true = tf.where(y_true==self.class_id, 1., 0.) # (B, S)
    tp = tf.reduce_sum(class_preds*class_true, axis=-1) # (B,)
    pred_p = tf.reduce_sum(class_preds, axis=-1) # (B,)
    self.precision.assign(
        (self.precision*self.i+tf.reduce_mean(special_divide(tp,pred_p)))/(self.i+1)
    )
    self.i.assign_add(1)

  def result(self):
    return self.precision


class SparseRecall(tf.keras.metrics.Metric):
  def __init__(self, name="recall", class_id=1, batch_size=64, **kwargs):
    super().__init__(name=name, **kwargs)
    self.recall = self.add_weight(name="recall", initializer="zeros", dtype=tf.float32)
    self.i = self.add_weight(name="i", initializer="zeros")
    self.class_id = class_id

  def update_state(self, y_true, y_pred, sample_weight=None):
    """
      y_true assumed (Batch, Seq)
      y_pred assumed (Batch, Seq, Classes) where Classes is a distribution on classes

      Assumes fixed and constant batch size
      Convert distribution of y_pred to prediction via argmax, thus can also handle logits.
    """
    preds = tf.argmax(y_pred, axis=-1) # (B, S)
    class_preds = tf.where(preds==self.class_id, 1., 0.) # (B, S)
    class_true = tf.where(y_true==self.class_id, 1., 0.) # (B, S)
    tp = tf.reduce_sum(class_preds*class_true, axis=-1) # (B,)
    orig_p = tf.reduce_sum(class_true, axis=-1) # (B,)
    self.recall.assign(
        (self.recall*self.i+tf.reduce_mean(special_divide(tp,orig_p)))/(self.i+1)
    )
    self.i.assign_add(1)

  def result(self):
    return self.recall

class SparseAccuracyWithIgnore(tf.keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name="sparse_categorical_accuracy", dtype=None, ignore_token=None):
        self.ignore_token = ignore_token
        super().__init__(name, dtype)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.ignore_token is not None:
            mask = tf.math.not_equal(y_true, tf.cast(self.ignore_token, y_true.dtype))
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)
            # TODO: Confirm the masking semantics do as we expect
            # bit weird it doesn't return a ragged tensor
            # Also a bit weird it wasn't a problem before
        return super().update_state(y_true, y_pred)

class SparseF1(tf.keras.metrics.Metric):
    def __init__(self, name=None, class_id=None, **kwargs):
        super().__init__(name, **kwargs)
        self.precision = SparsePrecision(class_id=class_id)
        self.recall = SparseRecall(class_id=class_id)
    
    def update_state(self, *args, **kwargs):
        self.precision.update_state(*args, **kwargs)
        self.recall.update_state(*args, **kwargs)
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        
    def result(self):
        return 2*(self.precision.result()*self.recall.result()/(self.precision.result()+self.recall.result()))