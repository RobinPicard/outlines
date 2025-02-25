from outlines.generate.generator import ModelGenerator, Generator
from outlines.types import Regex


def regex(model, regex_str: str | Regex) -> ModelGenerator:
    """Generate structured text in the language of a regular expression.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    regex_str:
        The regular expression that the output must follow.

    Returns
    -------
    A `ModelGenerator` instance that generates text.

    """
    if isinstance(regex_str, Regex):
        regex_str = regex_str.pattern

    return Generator(model, Regex(regex_str))
