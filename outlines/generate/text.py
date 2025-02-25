from outlines.generate.generator import ModelGenerator, Generator


def text(model) -> ModelGenerator:
    """Generate text with a model.

    Arguments
    ---------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.

    Returns
    -------
    A `ModelGenerator` instance that generates text.

    """
    return Generator(model)
