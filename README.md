# model-compression
Tools for compressing deep networks

# Installation
pip install -r requirements.txt
See deepcompression.ipynb for usage examples

# Notes
* **pruned_layers.py** contains the pruning of DNNs to reduce the storage of insignificant weight parameters with 2 methods: pruning by percentage and prune by standara deviation.
* **train_util.py** includes the training process of DNNs with pruned connections.
* **quantize.py** applies the quantization (weight sharing) part on the DNN to reduce the storage of weight parameters.
* **huffman_coding.py** applies the Huffman coding onto the weight of DNNs to further compress the weight size.

Creates the following files:
* **net_before_pruning.pt** is the weight parameters before applying pruning on DNN weight parameters.
* **net_after_pruning.pt** is the weight paramters after applying pruning on DNN weight parameters.
* **net_after_quantization.pt** is the weight parameters after applying quantization (weight sharing) on DNN weight parameters.
* **codebook_vgg16.npy** is the quantization codebook of each layer after applying quantization (weight sharing).
* **huffman_encoding.npy** is the encoding map of each item within the quantization codebook in the whole DNN architecture.
* **huffman_freq.npy** is the frequency map of each item within the quantization codebook in the whole DNN. 