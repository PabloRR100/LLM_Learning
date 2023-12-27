"""
"""

import os
from pathlib import Path

from mingpt.model import GPT
from mingpt.servable import GPTServable
from mingpt.utils import compare_weights


# ROOT = Path(__file__).parent.parent
ROOT = Path(__file__).parent.parent.parent
os.environ.setdefault("LOG_LEVEL", "DEBUG")


default_model = GPT.load_default()
default_config = default_model.config
assert default_config.to_dict() == default_model.config.to_dict()

servable = GPTServable(name="gpt-2", model=default_model, device="cpu")
def_output = servable.run_inference(prompt="Tell me a joke: ")

# ---
# Regular model
# ---

tmp = ROOT / "models/"
path = str(tmp / "minGPT")

servable.export(path=path)
servable = GPTServable.load(path=path, config=default_config, name="gpt-2")
assert compare_weights(servable.model, default_model)

load_output = servable.run_inference(prompt="Tell me a joke: ")

# ---
# JIT model
# ---

# jit_path = str(tmp / "minGPT")
# servable.export_jit(path=jit_path)
# servable_jit = GPTServable.load_jit(path=jit_path, name="gpt-2")
# jit_output = servable_jit.run_jit_inference(prompt="Tell me a joke: ")

# ---
# ONNX model
# ---

onnx_path = str(tmp / "minGPT")
servable.export_onnx(path=onnx_path)
servable_onnx = GPTServable.load_onnx(path=onnx_path)
onnx_output = servable_onnx.run_onnx_inference(prompt="Tell me a joke: ")

# ---
# Outputs comparison
# ---

print(def_output[0])
print(load_output[0])
# print(jit_output[0])
print(onnx_output[0])
assert def_output[0] == load_output[0]
# assert def_output[0] == jit_output[0]
assert def_output[0] == onnx_output[0]

