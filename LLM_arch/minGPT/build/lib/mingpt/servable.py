"""
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from mingpt.model import GPTModel
from mingpt.bpe import BPETokenizer
from mingpt.utils import timing


Input = str | list[str]
LOG = logging.getLogger(__name__)


@dataclass
class Servable(ABC):

    name: str
    device: str
    model: nn.Module

    @abstractmethod
    def pre_process(self, prompt: Input) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def post_process(self, x: torch.Tensor) -> Input:
        raise NotImplementedError

    @abstractmethod
    def run_inference(self, prompt: Input) -> Input:
        raise NotImplementedError


@dataclass
class GPTServable(Servable):

    name: str
    model: GPTModel
    tokenizer: BPETokenizer = BPETokenizer()

    def __post_init__(self):
        self.model.to(self.device)
        self.model.eval()

    def pre_process(self, prompt: Input):
        if prompt == '':
            # to create unconditional samples...
            # manually create a tensor with only the special <|endoftext|> token
            # similar to what openai's code does
            # here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = torch.tensor([[self.tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
        else:
            x = self.tokenizer(prompt).to(self.device)
        return x

    def post_process(self, preds, num_samples=10) -> Input:
        outputs = []
        for i in range(num_samples):
            out = self.tokenizer.decode(preds[i].cpu().squeeze())
            outputs.append(out)
        return outputs

    # @timing
    def run_inference(
        self,
        prompt: Input,
        num_samples: int = 1,
        max_new_tokens=20,
        do_sample=False,
        top_k=40
    ) -> Input:
        embedded = self.pre_process(prompt)
        inferred = self.model.generate(embedded, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k)
        outputs = self.post_process(inferred, num_samples=num_samples)
        return outputs

    # @timing
    def export(self, path: str):
        torch.save(self.model.state_dict(), path)

    @classmethod
    # @timing
    def load(cls, path: str, config, name: str = None, device: str = "cpu") -> "GPTServable":
        config.model_type = None
        model = GPTModel(config)
        saved_state_dict = torch.load(path)
        model.load_state_dict(saved_state_dict)
        model.to(device)
        model.eval()
        return cls(model=model, device=device, name=name)

    # @timing
    def export_jit(self, path: str):
        traced_script_module = torch.jit.script(self.model)
        traced_script_module.save(path)
        torch.jit.save(traced_script_module, path)

    @classmethod
    # @timing
    def load_jit(cls, path: str, name: str = None, device: str = "cpu") -> "GPTServable":
        model = torch.jit.load(path)
        model.to(device)
        model.eval()
        return cls(model=model, device=device, name=name)


if __name__ == "__main__":

    import os
    from pathlib import Path
    from utils import compare_weights

    ROOT = Path(__file__).parent.parent
    os.environ.setdefault("LOG_LEVEL", "DEBUG")

    default_model = GPTModel.load_default()
    default_config = default_model.config
    assert default_config.to_dict() == default_model.config.to_dict()

    servable = GPTServable(name="gpt-2", model=default_model, device="cpu")
    def_output = servable.run_inference(prompt="Tell me a joke: ")

    # Regular model
    # ---

    tmp = ROOT / "tmp"
    path = str(tmp / "gpt.pt")

    servable.export(path=path)
    servable = GPTServable.load(path=path, config=default_config, name="gpt-2")
    assert compare_weights(servable.model, default_model)

    load_output = servable.run_inference(prompt="Tell me a joke: ")

    # JIT model
    # ---

    jit_path = str(tmp / "gpt_torch_script.pt")
    servable.export_jit(path=jit_path)
    servable_jit = GPTServable.load_jit(path=jit_path, name="gpt-2")
    jit_output = servable_jit.run_inference(prompt="Tell me a joke: ")

    # Outputs comparison
    # ---

    print(def_output[0])
    print(load_output[0])
    print(jit_output[0])
    assert def_output[0] == load_output[0]
    assert def_output[0] == load_output[0]


