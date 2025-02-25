import json as pyjson
from enum import Enum
from typing import Optional, Union
from typing_extensions import _TypedDictMeta

from pydantic import BaseModel

from outlines.fsm.json_schema import get_schema_from_enum, get_schema_from_signature
from outlines.generate.generator import ModelGenerator, Generator
from outlines.types import JsonType

from .regex import regex


def json(
    model,
    schema_object: Union[str, type(BaseModel), dict, _TypedDictMeta],
    whitespace_pattern: Optional[str] = None,
) -> ModelGenerator:
    """
    Generate structured JSON data with a `Transformer` model based on a specified JSON Schema.

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    schema_object:
        The JSON Schema to generate data for. Can be a JSON Schema
        specification represented as a string or a dictionary, or a Pydantic
        BaseModel.
    whitespace_pattern
        Pattern to use for JSON syntactic whitespace (doesn't impact string literals)
        Example: allow only a single space or newline with `whitespace_pattern=r"[\n ]?"`

    Returns
    -------
    A `ModelGenerator` instance that generates text.

    """
    return Generator(model, JsonType(schema_object, whitespace_pattern))
