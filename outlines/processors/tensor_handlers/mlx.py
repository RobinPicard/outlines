from outlines.processors.tensor_handlers.base import TensorHandler


class MLXTensorHandler(TensorHandler):
    def __init__(self):
        import mlx

        self.mlx = mlx

    @property
    def library_name(self):
        return "mlx"

    def shape(self, tensor):
        return tensor.shape

    def unsqueeze(self, tensor):
        return self.mlx.expand_dims(tensor, axis=0)

    def squeeze(self, tensor):
        return self.mlx.squeeze(tensor, axis=0)

    def to_list(self, tensor):
        return tensor.tolist()

    def to_scalar(self, tensor):
        return tensor.item()

    def full_like(self, tensor, fill_value):
        return self.mlx.full_like(tensor, fill_value)

    def concatenate(self, tensors):
        return self.mlx.concatenate(tensors, axis=0)

    def get_device(self, tensor):
        return None

    def to_device(self, tensor, device):
        return tensor

    def boolean_ones_like(self, tensor):
        return self.mlx.ones_like(tensor, dtype=self.mlx.bool_)

    def apply_mask(self, tensor, mask, value):
        result = tensor.copy()
        result[mask] = value
        return result
