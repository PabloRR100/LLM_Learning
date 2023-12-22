# LLMs 

Personal repository to enhance LLMs understanding.

## Contents

### MinGPT

- https://github.com/karpathy/minGPT

A PyTorch re-implementation of GPT, both training and inference.  
minGPT tries to be small, clean, interpretable and educational. 

The minGPT library is three files: 
- mingpt/model.py contains the actual Transformer model definition
- mingpt/bpe.py contains a mildly refactored Byte Pair Encoder that translates between text and sequences of integers exactly like OpenAI did in GPT
- mingpt/trainer.py is (GPT-independent) PyTorch boilerplate code that trains the model  

Then there are a number of demos and projects that use the library in the projects folder

- projects/adder --> trains a GPT from scratch to add numbers
- projects/chargpt --> trains a GPT to be a character-level language model on some input text file
- demo.ipynb --> shows a minimal usage of the GPT and Trainer in a notebook format on a simple sorting example
- generate.ipynb --> shows how one can load a pretrained GPT2 and generate text given some prompt

### NanoGPT

- https://github.com/karpathy/nanoGPT

Simple codebase for training/finetuning medium-sized GPTs.  
It is a rewrite of minGPT that prioritizes teeth over education


### GigaGPT

- https://github.com/Cerebras/gigaGPT

We present gigaGPT – the simplest implementation for training large language models with tens or hundreds of billions of parameters. This work was inspired by Andrej Karpathy's nanoGPT. However, while nanoGPT is designed to train medium sized models up to around the 1B parameter range, gigaGPT leverages Cerebras hardware to use a single simple model definition and training loop to scale to GPT-3 sized models run across exaflop scale clusters.

### GPT-Fast

- https://github.com/pytorch-labs/gpt-fast
- https://pytorch.org/blog/accelerating-generative-ai-2/

(PyTorch Official)  
*Simple and efficient pytorch-native transformer text generation.*

A minimalistic, PyTorch-only decoding implementation loaded with best practices: int8/int4 quantization, speculative decoding, Tensor parallelism, etc. Boosts the "clock speed" of LLM OS by 10x with no model change! 


### LIT-GPT

- https://github.com/Lightning-AI/lit-gpt


(PyTorch Lightning Official)  



### LLM-Visualization

- https://bbycroft.net/llm


