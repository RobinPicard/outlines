import pytest
from pytest import mark
import torch
import numpy as np
import jax
import jax.numpy as jnp

from outlines.processors.tensor_handlers import (
    NumpyTensorHandler,
    TorchTensorHandler,
    MLXTensorHandler,
    JAXTensorHandler
)

try:
    import mlx_lm
    import mlx.core as mx

    HAS_MLX = mx.metal.is_available()
except ImportError:
    HAS_MLX = False


handlers = {
    "numpy": NumpyTensorHandler(),
    "torch": TorchTensorHandler(),
    "jax": JAXTensorHandler(),
}
if HAS_MLX:
    handlers["mlx"] = MLXTensorHandler()

frameworks = list(handlers.keys())

def create_tensor(framework, shape, dtype=None):
    if framework == "torch":
        return torch.randn(*shape)
    elif framework == "numpy":
        return np.random.randn(*shape)
    elif framework == "jax":
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape=shape)
    elif framework == "mlx" and HAS_MLX:
        return mx.random.randn(*shape)

def compare_tensors(framework, tensor1, tensor2):
    if framework == "torch":
        return torch.allclose(tensor1, tensor2)
    elif framework == "numpy":
        return np.array_equal(tensor1, tensor2)
    elif framework == "jax":
        return jax.numpy.array_equal(tensor1, tensor2)
    elif framework == "mlx" and HAS_MLX:
        return mx.ndarray.array_equal(tensor1, tensor2)


@pytest.mark.parametrize("framework", frameworks)
def test_tensor_handler_shape(framework):
    # 1d tensor
    tensor_1d = create_tensor(framework, (2,))
    result_1d = handlers[framework].shape(tensor_1d)
    assert len(result_1d) == 1
    assert result_1d[0] == 2

    # 2d tensor
    tensor_2d = create_tensor(framework, (2, 3))
    result_2d = handlers[framework].shape(tensor_2d)
    assert len(result_2d) == 2
    assert result_2d[0] == 2
    assert result_2d[1] == 3

    # 3d tensor
    tensor_3d = create_tensor(framework, (2, 2, 3))
    result_3d = handlers[framework].shape(tensor_3d)
    assert len(result_3d) == 3
    assert result_3d[0] == 2
    assert result_3d[1] == 2
    assert result_3d[2] == 3


@pytest.mark.parametrize("framework", frameworks)
def test_tensor_handler_unsqueeze(framework):
    # 1d tensor
    tensor_1d = create_tensor(framework, (2,))
    result_1d = handlers[framework].unsqueeze(tensor_1d)
    assert result_1d.shape == (1, 2)

    # 2d tensor
    tensor_2d = create_tensor(framework, (2, 3))
    result_2d = handlers[framework].unsqueeze(tensor_2d)
    assert result_2d.shape == (1, 2, 3)


@pytest.mark.parametrize("framework", frameworks)
def test_tensor_handler_squeeze(framework):
    # 1d tensor
    tensor_1d = create_tensor(framework, (1,))
    result_1d = handlers[framework].squeeze(tensor_1d)
    with pytest.raises(TypeError):
        len(result_1d)

    # 2d tensor
    tensor_2d = create_tensor(framework, (1, 2))
    result_2d = handlers[framework].squeeze(tensor_2d)
    assert result_2d.shape == (2,)

    # 3d tensor
    tensor_3d = create_tensor(framework, (1, 2, 3))
    result_3d = handlers[framework].squeeze(tensor_3d)
    assert result_3d.shape == (2, 3)


@pytest.mark.parametrize("framework", frameworks)
def test_tensor_handler_to_list(framework):
    # 1d tensor
    tensor_1d = create_tensor(framework, (2,))
    result_1d = handlers[framework].to_list(tensor_1d)
    assert isinstance(result_1d, list)
    assert len(result_1d) == 2

    # 2d tensor
    tensor_2d = create_tensor(framework, (2, 3))
    result_2d = handlers[framework].to_list(tensor_2d)
    assert isinstance(result_2d, list)
    assert len(result_2d) == 2
    assert len(result_2d[0]) == 3
    assert len(result_2d[1]) == 3

    # 3d tensor
    tensor_3d = create_tensor(framework, (2, 2, 3))
    result_3d = handlers[framework].to_list(tensor_3d)
    assert isinstance(result_3d, list)
    assert len(result_3d) == 2
    assert len(result_3d[0]) == 2
    assert len(result_3d[1]) == 2
    assert len(result_3d[0][0]) == 3
    assert len(result_3d[0][1]) == 3
    assert len(result_3d[1][0]) == 3
    assert len(result_3d[1][1]) == 3


@pytest.mark.parametrize("framework", frameworks)
def test_tensor_handler_to_scalar(framework):
    # multi-elements tensor, should raise an error
    tensor_multi = create_tensor(framework, (2, 3))
    if framework == "torch":
        with pytest.raises(RuntimeError):
            handlers[framework].to_scalar(tensor_multi)
    else:
        with pytest.raises(ValueError):
            handlers[framework].to_scalar(tensor_multi)

    # single-element tensor
    tensor_single = create_tensor(framework, (1, 1))
    scalar = handlers[framework].to_scalar(tensor_single)
    assert isinstance(scalar, float)


@pytest.mark.parametrize("framework", frameworks)
def test_tensor_handler_full_like(framework):
    tensor = create_tensor(framework, (2, 3))
    result = handlers[framework].full_like(tensor, 0)
    assert result.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert result[i, j] == 0


@pytest.mark.parametrize("framework", frameworks)
def test_tensor_handler_concatenate(framework):
    # 1d tensors
    tensor1 = create_tensor(framework, (2,))
    tensor2 = create_tensor(framework, (2,))
    result = handlers[framework].concatenate([tensor1, tensor2])
    assert result.shape == (4,)
    assert result[0] == tensor1[0]
    assert result[1] == tensor1[1]
    assert result[2] == tensor2[0]
    assert result[3] == tensor2[1]

    # 2d tensors
    tensor1 = create_tensor(framework, (2, 3))
    tensor2 = create_tensor(framework, (2, 3))
    result = handlers[framework].concatenate([tensor1, tensor2])
    assert result.shape == (4, 3)
    for i in range(2):
        for j in range(3):
            assert result[i, j] == tensor1[i, j]
            assert result[i + 2, j] == tensor2[i, j]

    # 3d tensors
    tensor1 = create_tensor(framework, (2, 2, 3))
    tensor2 = create_tensor(framework, (2, 2, 3))
    result = handlers[framework].concatenate([tensor1, tensor2])
    assert result.shape == (4, 2, 3)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                assert result[i, j, k] == tensor1[i, j, k]
                assert result[i + 2, j, k] == tensor2[i, j, k]


@pytest.mark.parametrize("framework", frameworks)
def test_tensor_handler_to_device(framework):
    tensor = create_tensor(framework, (2, 3))
    device_tensor = handlers[framework].to_device(tensor, "cpu")

    if framework == "torch":
        assert device_tensor.device.type == "cpu"
        assert compare_tensors(framework, device_tensor, tensor)
    else:
        assert compare_tensors(framework, device_tensor, tensor)


@pytest.mark.parametrize("framework", frameworks)
def test_tensor_handler_boolean_ones_like(framework):
    tensor = create_tensor(framework, (2, 3))
    ones = handlers[framework].boolean_ones_like(tensor)

    assert ones.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert ones[i, j]


@pytest.mark.parametrize("framework", frameworks)
def test_tensor_handler_apply_mask(framework):
    tensor = create_tensor(framework, (2, 3))

    if framework == "torch":
        mask = torch.randn(2, 3) > 0
    elif framework == "numpy":
        mask = np.random.randn(2, 3) > 0
    elif framework == "jax":
        key = jax.random.PRNGKey(0)
        mask = jax.random.normal(key, shape=(2, 3)) > 0
    elif framework == "mlx" and HAS_MLX:
        mask = mx.random.randn(2, 3) > 0

    masked = handlers[framework].apply_mask(tensor, mask, float("-inf"))

    assert masked.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            if mask[i, j]:
                assert masked[i, j] == float("-inf")
            else:
                assert masked[i, j] == tensor[i, j]
