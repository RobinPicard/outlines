from .jax import JAXTensorHandler
from .mlx import MLXTensorHandler
from .numpy import NumpyTensorHandler
from .torch import TorchTensorHandler


def get_vector_handler(tensor_library_name: str):
    if tensor_library_name == "torch":
        return TorchTensorHandler()
    elif tensor_library_name == "mlx":
        return MLXTensorHandler()
    elif tensor_library_name == "jax":
        return JAXTensorHandler()
    elif tensor_library_name == "numpy":
        return NumpyTensorHandler()
    else:
        raise ValueError(f"Unsupported tensor library: {tensor_library_name}")
