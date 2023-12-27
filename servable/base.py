import json
from pathlib import Path
from abc import ABC, abstractmethod

from servable.schema import ServableSchema


class BaseServable(ABC):
    """
    A concrete class inheriting from BaseServable defines a servable
    pipeline that is loaded up from the LITServableLoader.
    """

    SCHEMA_SLUG = "schema.json"

    MODEL_TYPE: str
    MODEL_VARIANT: str

    @property
    def is_config_driven(self):
        """
        Track whether the given servable is config driven or not.
        """
        return False

    @classmethod
    def _export_schema(cls, schema, servable_path):
        """
        Helper to export schemas to the appropriate place within model_path.

        Args:
            schema (ServableSchema): Schema of the servable that is being saved
            servable_path (str): Prefix (S3 or filesystem) from which to save the
                models schema.json file.

        Returns:
            None: Side-effect is that the schema is saved to model_path.
        """
        schema_path = str(Path(servable_path) / cls.SCHEMA_SLUG)

        with open(schema_path, mode="w") as schema_f:
            json.dump(schema.to_dict(), schema_f, indent=4)

    @classmethod
    def load_servable_schema(cls, servable_path=None) -> "ServableSchema":
        """
        Helper to load a schema from a model_path.

        N.B., Calls to this method are memoize-d to avoid repeated calls to S3
        in the case that model_path is an S3 path.

        Args:
            servable_path (str): Prefix (S3 or filesystem) from which to load the
                schema.

        Returns:
            ServableSchema: Model schema loaded from model_path.
        """
        schema_path = str(Path(servable_path) / cls.SCHEMA_SLUG)
        with open(schema_path, mode="r") as schema_f:
            return ServableSchema.from_dict(json.load(schema_f))

    def to_accelerator_if_available(self):
        """
        Moves a servable onto an (unspecified) computation accelerator (for
        example, a GPU) if the accelerator is the accelerator is available.

        Implementation and logic are left to concrete subclasses.

        Returns:
            BaseServable: self
        """
        return self

    def to_cpu(self):
        """
        Moves a servable onto the CPU.

        Implementation and logic are left to concrete subclasses.

        Returns:
            BaseServable: self
        """
        return self

    @abstractmethod
    def run_inference(self, *args, **kwargs):
        """Run the inference pipeline for a concrete subclass.

        Each subclass is expected to implement this method, where the
        expectation is that this method will operate on a batch of inputs and
        return a batch of outputs.

        Args:
            *args, **kwargs: Inputs required for the concrete subclass.
        """
        pass

    @abstractmethod
    def export(self, servable_path) -> "ServableSchema":
        """
        Exports the Servable to a particular prefix (i.e., model artifacts may
        live at {model_path}/foo.tgz, etc.)

        Args:
            servable_path (str): The prefix to save the servable to. This can be
            on S3 (prefixed with s3://) or on the local filesystem.

        Returns:
            ServableSchema: The schema of the servable saved out.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, servable_path, schema):
        """
        Loads a servable from a servable_path based on the parameters contained in a
        schema. This method is supposed to be used by loader-type objects such as
        LITServableLoader, which already have the schema read from the servable path.
        To load a servable directly from the path, use load_from_path().

        Args:
            servable_path (string): Path to the servable to load
            schema (ServableSchema): Schema containing relevant parameters to
                load the servable.

        Returns:
            BaseServable: Concrete implementation of a BaseServable.
        """
        pass

    @classmethod
    def load_from_path(cls, servable_path):
        """
        Loads a servable based on a servable path.

        Internally, will rebuild the schema to access all relevant info to
        rebuild the servable.

        Args:
            servable_path (str): Prefix (S3 or filesystem) from which to load the
                servable.

        Returns:
            BaseServable: Concrete implementation of a BaseServable.
        """
        return cls.load(
            servable_path, schema=cls.load_servable_schema(servable_path=servable_path)
        )
