from transformer_segmentation import MAX_CHARS
from transformer_segmentation import BATCH_SIZE
from transformer_segmentation import D_MODEL
from transformer_segmentation import NUM_LAYERS
from transformer_segmentation import NUM_ATTENTION_HEADS
from transformer_segmentation import DFF

precision = 32 # bit
gigabyte_conversion = int(10e9)

simply_data = BATCH_SIZE*MAX_CHARS*D_MODEL
projection_kernels = 3*D_MODEL*D_MODEL*NUM_ATTENTION_HEADS # Could start using smaller key dim
projection_results = 3*MAX_CHARS*D_MODEL*BATCH_SIZE*NUM_ATTENTION_HEADS
dot_prod_results = BATCH_SIZE*NUM_ATTENTION_HEADS*MAX_CHARS*MAX_CHARS
output_proj_kernels = MAX_CHARS*D_MODEL*NUM_ATTENTION_HEADS
output_proj_results = BATCH_SIZE*NUM_ATTENTION_HEADS*MAX_CHARS*D_MODEL
ff_kernel = D_MODEL*DFF
ff_results = BATCH_SIZE*NUM_ATTENTION_HEADS*DFF*D_MODEL

attention_mechanism = sum([
    projection_kernels, projection_results,
    dot_prod_results,
    output_proj_kernels, output_proj_results,
    ])

encoder_layer = sum([
    attention_mechanism,
    ff_kernel, ff_results,
])

decoder_layer = sum([
    attention_mechanism,
    attention_mechanism,
    ff_kernel, ff_results,
])

encoder = NUM_LAYERS*encoder_layer
decoder = NUM_LAYERS*decoder_layer

total_mem = simply_data+encoder+decoder

print(
f"Input embedded: {simply_data*precision/gigabyte_conversion:.2f}GB ({simply_data/total_mem:.2f})\n"
f"Single attention mechanism: {attention_mechanism*precision/gigabyte_conversion:.2f}GB ({attention_mechanism/total_mem:.2f})\n"
f"Encoder layer: {encoder_layer*precision/gigabyte_conversion:.2f}GB ({encoder_layer/total_mem:.2f})\n"
f"Decoder layer: {decoder_layer*precision/gigabyte_conversion:.2f}GB ({decoder_layer/total_mem:.2f})\n"
f"Encoder overall: {encoder*precision/gigabyte_conversion:.2f}GB ({encoder/total_mem:.2f})\n"
f"Decoder overall: {decoder*precision/gigabyte_conversion:.2f}GB ({decoder/total_mem:.2f})\n"
f"Overall: {total_mem*precision/gigabyte_conversion:.2f}GB\n"
f"Params approx: {total_mem:,}"
)