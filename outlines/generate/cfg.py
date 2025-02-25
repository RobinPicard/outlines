from outlines.generate.generator import ModelGenerator, Generator
from outlines.types import CFG


def cfg(model, cfg_str: str) -> ModelGenerator:
    """Generate text in the language of a Context-Free Grammar

    Arguments
    ---------
    model:
        An `outlines.model` instance.
    cfg_str:
        A string representation of the Context-Free Grammar.

    Returns
    -------
    A `ModelGenerator` instance that generates text.

    """
    return Generator(model, CFG(cfg_str))
