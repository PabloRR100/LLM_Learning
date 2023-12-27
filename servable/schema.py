import dataclasses
from types import SimpleNamespace


class ExtraSchema(SimpleNamespace):
    pass


@dataclasses.dataclass
class ServableSchema:
    """
    A ServableSchema defines the parameters we consider to be sufficient to run inference on a model.

    Attributes:

        model_type: The type of the model that is served by the given servable.
        model_variant: An internal-facing identifier of what type of model is
            under the covers. For example, for a text formatter, having
            model_type == ModelType.TEXT_FORMATTER and
            model_variant == "bert-base-uncased" informs us we need to use the
            BertPunctuatorServable object to run inference. Though the semantics
            are enforced, model_type serves as the 'task', and model_variant
            serves as the 'model choice'.
        extra: Extra key-value pairs held in a namespace-like object to store
            other as-needed info. For example, a sentiment classification model
            may have `classes=['positive', 'negative']` to help clients know
            the label spaces. The expectation is that everything in extra is
            propagated to model depot.
    """

    model_type: str
    model_variant: str
    extra: ExtraSchema = dataclasses.field(default_factory=ExtraSchema)

    def to_dict(self):
        """
        Converts a ServableSchema object to a dict

        Returns:
            dict: dictionary repr of the schema
        """
        dict_repr = dict(self.__dict__)
        dict_repr["model_type"] = dict_repr["model_type"].value
        dict_repr["extra"] = dict(dict_repr["extra"].__dict__)
        return dict_repr

    @classmethod
    def from_dict(cls, d):
        try:
            model_type = d["model_type"]
            model_variant = d["model_variant"]
            if not isinstance(model_variant, str):
                raise TypeError("model_variant must be a string")

            cct_model = d.get("cct_model")

            extra = d.get("extra", {})
            if not isinstance(extra, dict):
                raise TypeError("extra must be a dictionary")

            extra = ExtraSchema(**extra)

            return cls(
                model_type=model_type,
                model_variant=model_variant,
                extra=extra,
            )
        except TypeError as _:
            raise
        except Exception as err:
            raise RuntimeError(f"Failed to parse {cls.__qualname__}") from err
