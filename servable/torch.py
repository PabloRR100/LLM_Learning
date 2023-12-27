import logging
from pathlib import Path
from functools import partial

import torch
import torch.jit

from servable.base import BaseServable
from servable.device import get_auto_device


LOG = logging.getLogger(__name__)


class TorchServable(BaseServable):

    MODEL_SLUG = "model.pt"

    def __init__(self, model, device=None):
        """
        Construct a Servable object for Torch models.

        Note: this does not handle any training or setup, but assumes you've
        trained the model separately.

        Args:
            model (nn.Module): Trained model to be serialized.
            device (Optional[torch.device]): Device to move the model to.
        """
        self.device = device or get_auto_device()
        self.model = model.eval().to(self.device)

    def to_accelerator_if_available(self):
        self.model.to(device=self.device)
        return self

    def to_cpu(self):
        self.model.to(device=torch.device("cpu"))
        return self

    @classmethod
    def _export_model_and_schema(cls, model_instance, schema, servable_path):
        """
        Helper to export torch models and schemas to the appropriate places
        within model_path.

        Args:
            model_instance (torch.nn.Module): Model to export. Will detect a
                regular module versus a JIT module and save accordingly.
            schema (ServableSchema): Schema of the servable that is being saved
            servable_path (str): Prefix (S3 or filesystem) from which to load the
                model's state dict.

        Returns:
            None: Side-effect is that the model and schema are saved to
                model_path.
        """
        cls._export_schema(schema, servable_path=servable_path)
        model_object_path = str(Path(servable_path) / cls.MODEL_SLUG)

        with open(model_object_path, mode="w") as f:
            if isinstance(model_instance, torch.jit.ScriptModule):
                torch.jit.save(model_instance, f.name)
            else:
                torch.save(model_instance.state_dict(), f.name)


class TorchJITServable(TorchServable):
    def __init__(self, model, device=None, jit_fn=None):
        """
        Instantiate a new JIT servable.

        Args:
            model (nn.Module): Torch model to wrap.
                The model will be JIT compiled at instantiation for
                portability, if the model is not already a ScriptModule.
            device (Optional[torch.device]): Device to run the model on.
            jit_fn (Optional[Callable[torch.nn.Module, torch.nn.ScriptModule]]):
                If None, will select an appropriate JIT conversion function
                based on the presence of a `trace_example_input` attribute on
                the model. Else, expected to be a function that takes a single
                argument, the model, and produces a JIT model.
        """

        try:
            if not isinstance(model, torch.jit.ScriptModule):
                trace_example_input = getattr(model, "trace_example_input", None)
                # If we have defined the trace_example_input property, use that
                # to trace the module. Else, just use scripting to convert.
                if not jit_fn:
                    if trace_example_input is not None:
                        jit_fn = partial(
                            torch.jit.trace, example_inputs=trace_example_input
                        )
                    else:
                        jit_fn = torch.jit.script
                model = jit_fn(model.eval())
        except Exception as err:
            raise RuntimeError("Failed to script-mode JIT compile model") from err
        if not isinstance(model, torch.jit.ScriptModule):
            raise TypeError(
                f"Expected converted model to be a ScriptModule, "
                f"got type: {type(model)}"
            )
        super().__init__(model=model.eval(), device=device)

    @classmethod
    def _load_jit_model(cls, servable_path, device=None):
        device = device or get_auto_device()
        with open(str(Path(servable_path) / cls.MODEL_SLUG), mode="r") as f:
            model = torch.jit.load(f.name, map_location=torch.device("cpu")).eval()
            model.to(device)
        return model
