# Qwen 2.5 JAX

JAX implementation for Qwen 2.5 0.5B inference with proper JIT compilation. This loads pretrained weights from HuggingFace and converts them into JAX pytrees. The implementation uses pure functions throughout instead of the usual class-based approach, so XLA can do its thing and speed up generation after the first token compiles.

The code is pretty straightforward, everything is in `inference.py`. It supports greedy decoding and temperature sampling, and the functional style makes it easy to see what's happening at each step. No frameworks, no complicated abstractions, just the model architecture implemented in a way that JAX likes.