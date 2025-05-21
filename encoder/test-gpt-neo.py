#!/usr/bin/env python3

# This script fails. I'm not sure it will ever work. Since gpt-neo is now PyTorch only, I will investigate rewriting with PyTorch only.
# Traceback (most recent call last):
#   File "/home/scott/git/roof12/public/5dgai-capstone/encoder/test-gpt-neo.py", line 19, in <module>
#     model_tf = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
#   File "/home/scott/.local/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 574, in from_pretrained
#     raise ValueError(
# ValueError: Unrecognized configuration class <class 'transformers.models.gpt_neo.configuration_gpt_neo.GPTNeoConfig'> for this kind of AutoModel: TFAutoModelForCausalLM.
# Model type should be one of BertConfig, CamembertConfig, CTRLConfig, GPT2Config, GPT2Config, GPTJConfig, MistralConfig, OpenAIGPTConfig, OPTConfig, RemBertConfig, RobertaConfig, RobertaPreLayerNormConfig, RoFormerConfig, TransfoXLConfig, XGLMConfig, XLMConfig, XLMRobertaConfig, XLNetConfig.


# Import necessary modules from the transformers library and TensorFlow
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TFAutoModelForCausalLM

import tensorflow as tf

# Define the model name (GPT-Neo with 125M parameters)
model_name = "EleutherAI/gpt-neo-125M"

# Load the GPT-Neo model in PyTorch format
# GPTNeoForCausalLM is the class for causal language modeling tasks (text generation) in PyTorch
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Convert the PyTorch model to TensorFlow format
# TFAutoModelForCausalLM is the TensorFlow-compatible class for causal language modeling tasks
# The 'from_pt=True' argument specifies that we are converting from a PyTorch model to TensorFlow
model_tf = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)

# Load the tokenizer associated with the GPT-Neo model
# GPT2Tokenizer is used here, as GPT-2 and GPT-Neo models share the same tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the padding token to be the same as the end-of-sequence token
# This is required for padding sequences during text generation or model inference
tokenizer.pad_token = tokenizer.eos_token

# Print a confirmation message to indicate that the model has been successfully loaded in TensorFlow
print("TensorFlow model loaded successfully!")
