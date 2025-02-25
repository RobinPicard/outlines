from enum import Enum
from typing import List, Union

from outlines.generate.generator import ModelGenerator, Generator
from outlines.types import Choice


def choice(model, choices: Union[List[str], type[Enum]]) -> ModelGenerator:
    return Generator(model, Choice(choices))
