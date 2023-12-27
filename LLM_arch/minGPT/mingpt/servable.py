"""
"""
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import numpy as np
import torch.nn as nn
import onnxruntime as ort

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT
from mingpt.utils import timing


Input = Output = str | list[str]
LOG = logging.getLogger(__name__)


@dataclass
class Servable(ABC):

    name: str
    device: str
    model: nn.Module

    @abstractmethod
    def pre_process(self, prompt: Input) -> torch.LongTensor:
        raise NotImplementedError

    @abstractmethod
    def post_process(self, x: torch.LongTensor) -> Output:
        raise NotImplementedError

    @abstractmethod
    def run_inference(self, prompt: Input) -> Output:
        raise NotImplementedError


@dataclass
class GPTServable(Servable):

    name: str
    model: GPT = None
    tokenizer: BPETokenizer = BPETokenizer()
    session: ort.InferenceSession = None

    def __post_init__(self):
        self.model.to(self.device)
        self.model.eval()

    def pre_process(self, prompt: Input) -> torch.LongTensor:
        if prompt == '':
            # to create unconditional samples...
            # manually create a tensor with only the special <|endoftext|> token
            # similar to what openai's code does
            # here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = torch.tensor([[self.tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
        else:
            x = self.tokenizer(prompt).long().to(self.device)
        return x

    def post_process(self, preds, num_samples=10) -> Output:
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
    ) -> Output:
        embedded = self.pre_process(prompt)
        inferred = self.model.generate(embedded, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k)
        outputs = self.post_process(inferred, num_samples=num_samples)
        return outputs

    # @timing
    def export(self, path: str):
        model_path = str(Path(path) / "model.pt")
        torch.save(self.model.state_dict(), model_path)

    @classmethod
    # @timing
    def load(cls, path: str, config, name: str = None, device: str = "cpu") -> "GPTServable":
        config.model_type = None
        model = GPT(config)
        model_path = Path(path) / "model.pt"
        if not model_path.parent.exists():
            raise ValueError(f"Model not found at: {model_path}")
        saved_state_dict = torch.load(model_path)
        model.load_state_dict(saved_state_dict)
        model.to(device)
        model.eval()
        return cls(model=model, device=device, name=name)

    # @timing
    def export_jit(self, path: str):
        # # opt 1
        # traced_script_module = torch.jit.script(self.model)
        # traced_script_module.save(path)
        # torch.jit.save(traced_script_module, path)
        # opt2: using torch.jit.trace
        embeddings = self.pre_process("This is a sample sentence").to(self.device)
        traced_script_module = torch.jit.trace(self.model, embeddings)
        traced_script_module.save(path)

    @classmethod
    # @timing
    def load_jit(cls, path: str, name: str = None, device: str = "cpu") -> "GPTServable":
        model = torch.jit.load(path)
        model.to(device)
        model.eval()
        return cls(model=model, device=device, name=name)

    def run_jit_inference(
        self,
        prompt: Input,
        num_samples: int = 1,
        max_new_tokens=20,
        do_sample=False,
        top_k=40
    ) -> Output:
        return self.run_inference(
            prompt=prompt, num_samples=num_samples, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k
        )

    # @timing
    def export_onnx(self, path: str):
        model_path = Path(path) / "model_onnx.pt"
        if not model_path.parent.exists():
            raise ValueError(f"Model not found at: {model_path}")
        self.model.eval()
        embeddings = self.pre_process("This is a sample sentence").to(self.device)
        export_output = torch.onnx.dynamo_export(self.model, embeddings, verbose=True)
        # export_output.model_proto
        export_output.save(str(model_path))

    @classmethod
    def load_onnx(cls, path: str, name: str = None, device: str = "cpu") -> "GPTServable":
        # Load the ONNX model
        session = ort.InferenceSession(path)
        return cls(
            model=None,
            device=device,
            name=name,
            session=session
        )

    def run_onnx_inference(
        self,
        prompt: Input,
        num_samples: int = 1,
        max_new_tokens=20,
        do_sample=False,
        top_k=40
    ) -> Output:
        # Get input and output names for the model
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        # Prepare the input data (adjust the shape and type according to your model's needs)
        # input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        input_data = self.pre_process(prompt).numpy()

        # Run inference
        dsa = self.session.run([output_name], {input_name: input_data})

