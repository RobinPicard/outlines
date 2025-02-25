from outlines.fsm.types import python_types_to_regex
from outlines.generate.generator import ModelGenerator, Generator
from outlines.types import Regex


def format(model, python_type) -> ModelGenerator:
    """Generate structured data that can be parsed as a Python type.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    python_type:
        A Python type. The output of the generator must be parseable into
        this type.

    Returns
    -------
    A `ModelGenerator` instance that generates text.

    """
    regex_str, format_fn = python_types_to_regex(python_type)
    return Generator(model, Regex(regex_str.pattern))
