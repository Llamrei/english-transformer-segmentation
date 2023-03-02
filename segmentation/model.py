import tensorflow as tf
import numpy as np

# TODO: Add logging

# # Embedding defn
def positional_encoding(length, depth):
    """
    Generates a matrix following:
    $$
        PE_{pos,i} = trig(\frac{pos, 10000^{\frac{i, d}})
    $$
    where d is the dimensionality of the output embedding and the position
    is defined absolutely (from 0).
    """
    per_trig_d_model = depth/2
    

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(per_trig_d_model)[np.newaxis, :]/per_trig_d_model   # (1, depth/2)
    angle_rates = 1 / (10000**depths)         # (1, depth/2)
    angle_rads = positions * angle_rates      # (seq, depth/2)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)], # (seq, depth)
      axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

def positional_encoding_alternating(length, depth):
    # TODO: Try this alternative encoding
    max_wavelength = 10000.

    pos = np.arange(length)
    inx = np.arange(depth)

    I, P = np.meshgrid(inx, pos)
    pe_even = np.sin(P / max_wavelength**(I/depth))
    pe_odd = np.cos(P / max_wavelength**(I/depth))
        
    pe = np.zeros((length, depth))
    pe[:, ::2] = pe_even[:, ::2]
    pe[:, 1::2] = pe_odd[:, ::2]
    return tf.constant(pe, dtype=tf.float32)

class PostionalEmbedding(tf.keras.layers.Layer):
    # TODO: FIX spelling
    def __init__(
            self,
            vocab_size, 
            d_model, 
            max_seq_len,
            pos_multiplier=1, 
            mask_zero=True):
        """
        Generate a layer to embed input tokens that are already int-encoded
         through a lookup embedding and positional information 
         (through additive positional embeddings as in Vaswani 2017).
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=mask_zero) 
        self.pos_encoding = positional_encoding(length=max_seq_len, depth=d_model)
        self.supports_masking = True
        self.pos_multiplier = pos_multiplier

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, mask=None):
        # Assumes (batch, seq_len) int-encoded inputs
        x = self.embedding(x) # (batch, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # TODO: try running without this
        x = x + self.pos_multiplier*self.pos_encoding[tf.newaxis, :, :] # new axis for batch dimension - try without
        return x

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
               dropout_rate=0.1,
               **kwargs
               ):
        super().__init__(**kwargs)


        # Multi-head self-attention.
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=d_model, # Size of each attention head for query Q and key K.
            dropout=dropout_rate, # TODO: maybe dropout?
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
            mask1 = mask[:, :, None] # (B, seq, 1)
            mask2 = mask[:, None, :] # (B, 1, seq)
            attention_mask = mask1 & mask2 
            # For each element in the sequence - what other elements can it attend to
            # only defined in this simple self-cartesion product because we are
            # simply attempting to mask away unused token slots for efficiency
            # Has block strucure, NOT upper triangular like a causal mask would
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
        ffn_output = self.dropout1(ffn_output, training=training) # TODO: try removing dropout errywhere
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
               tokenizer, # int-mode tokenizer for input text,
               seq_len,
               dropout_rate=0.1,
               pos_multiplier=1 
               ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Tokenization
        self.tokenizer = tokenizer

        # Embeddings + Positional encoding
        self.pos_embedding = PostionalEmbedding(tokenizer.vocabulary_size(), d_model, seq_len, pos_multiplier=pos_multiplier)

        # Encoder layers.
        self.enc_layers = [
            EncoderLayer(
              d_model=d_model,
              num_attention_heads=num_attention_heads,
              dff=dff,
              dropout_rate=dropout_rate,
              name=f"encoder_sublayer_{i}")
            for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def compute_mask(self, x, previous_mask=None):
        x = self.tokenizer(x)
        return self.pos_embedding.compute_mask(x, previous_mask)

    def call(self, x, training):
        # Sum up embeddings and positional encoding.
        x = self.tokenizer(x)
        mask = self.pos_embedding.compute_mask(x)
        # TODO: why am i giving back the mask here?
        # FIXME: Remove pointless mask passing
        x = self.pos_embedding(x, mask=mask)  # Shape `(batch_size, input_seq_len, d_model)`.
        # Add dropout.
        # ?
        x = self.dropout(x, training=training)

        # N encoder layers.
        for i in range(self.num_layers):
            # TODO: Do not need to pass mask 
            # TODO: Check mask propagation
            x = self.enc_layers[i](x, training, mask)

        return x  # Shape `(batch_size, input_seq_len, d_model)`.

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
               *,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               dropout_rate=0.1,
               **kwargs
               ):
        super().__init__(**kwargs)

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
        self_attn = self.layernorm1(attn_masked + x)

        # A boolean mask.
        attention_mask = None
        if mask is not None and enc_mask is not None:
            mask1 = mask[:, :, None]
            mask2 = enc_mask[:, None, :]
            attention_mask = mask1 & mask2

        # Multi-head cross-attention output (`tf.keras.layers.MultiHeadAttention `).
        attn_cross, attn_weights_cross = self.mha_cross(
            query=enc_output,
            key=self_attn,
            value=self_attn,
            attention_mask=attention_mask,  # A boolean mask that prevents attention to certain positions.
            return_attention_scores=True,  # Shape `(batch_size, num_queries, d_model)`.
            training=training  # A boolean indicating whether the layer should behave in training mode.
        )

        # Multi-head cross-attention output after layer normalization and a residual/skip connection.
        out2 = self.layernorm2(attn_cross + self_attn)  # (batch_size, source_seq_len, d_model)

        # Point-wise feed-forward network output.
        ffn_output = self.ffn(out2)  # Shape `(batch_size, source_seq_len, d_model)`.
        ffn_output = self.dropout1(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # Shape `(batch_size, source_seq_len, d_model)`.

        return out3


class Decoder(tf.keras.layers.Layer):
    output_tokens = [0, 1, 2]   # We only have a 3 token output language - space, no space, no text

    def __init__(self,
               *,
               num_layers,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               seq_len,
               dropout_rate=0.1
               ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PostionalEmbedding(len(Decoder.output_tokens), d_model, seq_len)

        self.dec_layers = [
            DecoderLayer(
              d_model=d_model,
              num_attention_heads=num_attention_heads,
              dff=dff,
              dropout_rate=dropout_rate,
              name=f"decoder_sublayer_{i}")
            for i in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def compute_mask(self, x, previous_mask=None):
        x = self.tokenizer(x)
        return self.pos_embedding.compute_mask(x, previous_mask)

    def call(self, dec_input, enc_output, enc_mask, training):
        mask = self.pos_embedding.compute_mask(dec_input)
        x = self.pos_embedding(dec_input)  # Shape: `(batch_size, target_seq_len, d_model)`.

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, mask, enc_output, enc_mask, training)

        # The shape of x is `(batch_size, target_seq_len, d_model)`.
        return x

class SpaceSegmentationTransformer(tf.keras.Model):
    """
    Transformer for finding space in english text without spaces

    Encoder takes in batches of strings, decoder takes in batches of seq_len
    tokens indicating space or not.

    """
    def __init__(self,
               *,
               num_layers, # Number of decoder layers.
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               input_tokenizer,
               seq_len,
               dropout_rate=0.1,
               classification_threshold=0.5,
               num_classes=2,
               pos_multiplier=1,
               ):
        super().__init__()
        d_model = d_model + 1 if d_model % 2 == 1 else d_model # Ensure even dimensionality so our positional encodings work
        self.encoder = Encoder(
          num_layers=num_layers,
          d_model=d_model,
          num_attention_heads=num_attention_heads,
          dff=dff,
          tokenizer=input_tokenizer,
          dropout_rate=dropout_rate,
          seq_len=seq_len,
          pos_multiplier=pos_multiplier
          )
        self.tokenizer = input_tokenizer
        self.dense = tf.keras.layers.Dense(num_classes, activation="softmax")  # Why does softmax here break it?

    def call(self, inputs, training=False):
        """
        Expects inputs of text to be segmentated and, if training, segmentation
        labels aligned to input text - where 2 indicates space, 1 indicates 
        no space and 0 indicates a missing character (essentially end token).

        Assumes batch first ranks
        """
        to_enc, to_dec = inputs
        run_sequential = False

        if not training:
            # This is where we will migrate sequential generation once we re-include the decoder
            batch_size = tf.shape(to_enc)[0]
            to_dec = tf.zeros((batch_size, 1))
            run_sequential = True

        
        # The encoder output.
        enc_output = self.encoder(to_enc, training)  # `(batch_size, inp_seq_len, d_model)`
        # enc_mask = self.encoder.compute_mask(to_enc)
        return self.dense(enc_output)

class LossWithVoids(tf.keras.losses.Loss):
    """
    Class to intercept any loss calculations when we want to ensure 'void'
    tokens in y_true are not included in loss calculations
    """
    def __init__(self, loss: tf.keras.losses.Loss, void_tokens: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.void_tokens = void_tokens

    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = tf.cast(tf.ones_like(y_true), tf.bool)
        for token in self.void_tokens:
            mask &= tf.not_equal(y_true, token) # (Batch, seq_len)
        return self.loss.__call__(y_true[mask], y_pred[mask]) 

class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=200):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return 2* tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)