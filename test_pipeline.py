import tensorflow as tf
import string
import html

# Data definition params 
JOINING_PUNC = r"([-'`])"
SPLITTING_PUNC = r'([!"#$%&()\*\+,\./:;<=>?@\[\\\]^_{|}~])'
NGRAM = 3
MAX_CHARS = 30
BATCH_SIZE = 16

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

def strip_spaces_and_set_predictions(text):
    x = tf.strings.lower(text)
    x = tf.strings.substr(x, 0, MAX_CHARS)
    # We want to sometimes replace punctuation with a space : "bad.sentence" -> "bad. sentence"
    # and other time to replace it with nothing: "don't" -> "dont"
    x = tf.strings.regex_replace(x, JOINING_PUNC, "")
    x = tf.strings.regex_replace(x, SPLITTING_PUNC, r"\1 ")   # \1 inserts captured splitting punctuation, raw so python doesnt magic it
    x = tf.strings.split(x)
    with open("test_result.pdf", "w") as f:
        f.write(str(x))
    no_whitespace_list = tf.strings.strip(x) # Remove any excess whitespace, e.g. tabs, double spaces etc.
    x = tf.strings.reduce_join(no_whitespace_list, separator="", axis=-1)
    y = tf.strings.reduce_join(no_whitespace_list, separator=" ", axis=-1)
    
    padding = tf.broadcast_to("#"*(NGRAM-1), (tf.shape(x)[0],) )
    x = tf.strings.reduce_join([padding, x], axis=0)
    y = tf.strings.reduce_join([padding, y], axis=0)
    
    final_boundary = tf.broadcast_to(" ", (tf.shape(x)[0],) )
    x = tf.strings.reduce_join([x, final_boundary], axis=0)
    y = tf.strings.reduce_join([y, final_boundary], axis=0)
    
    chars = tf.strings.unicode_split(y, "UTF-8")
    char_ngrams = tf.strings.ngrams(chars, ngram_width=NGRAM, separator="")
    with open("test_result.pdf", "a") as f:
        f.write(str(char_ngrams))
    labels = tf.strings.regex_full_match(char_ngrams, ".. ")
    labels = tf.cast(labels, 'int64')+1 # Add 1 to be able to tell difference between padding and prediction
    labels = labels.to_tensor(default_value=0, shape=[None, MAX_CHARS])
    # Roll so we predict for _future_ spaces not spaces in current token (if we knew space in current token we wouldnt have anything to solve)
    labels = tf.roll(labels, shift=-1, axis=-1)
    labels = labels[:,:-1]
    # labels = tf.random.experimental.stateless_shuffle(labels, seed=[1507, 1997]) # shuffles batch
    # labels = tf.random.stateless_binomial(shape=tf.shape(labels), seed=[1507, 1997], counts=1, probs=0.5)
    return (x, y), labels

print(
    strip_spaces_and_set_predictions(tf.constant([b"###amd debuts dualcore opteron processor amds new dualcore opteron chip is designed mainly for corporate computing applications, including databases, web services, and financial transactions. "])) 
    )