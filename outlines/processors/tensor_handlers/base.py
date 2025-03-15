from abc import ABC, abstractmethod


class TensorHandler(ABC):

    @property
    @abstractmethod
    def library_name(self):
        """Return the name of the library used by the tensor handler"""
        ...

    @abstractmethod
    def shape(self, tensor):
        """Get the shape of the tensor"""
        ...

    @abstractmethod
    def unsqueeze(self, tensor):
        """Add a dimension to the tensor at the specified position"""
        ...

    @abstractmethod
    def squeeze(self, tensor):
        """Remove a dimension from the tensor at the specified position"""
        ...

    @abstractmethod
    def to_list(self, tensor):
        """Convert the tensor to a list"""
        ...

    @abstractmethod
    def to_scalar(self, tensor):
        """Convert the tensor to a scalar"""
        ...

    @abstractmethod
    def full_like(self, tensor, fill_value):
        """Create a tensor with the same shape as the input tensor filled with a scalar value"""
        ...

    @abstractmethod
    def concatenate(self, tensors):
        """Concatenate a list of tensors along a specified dimension"""
        ...

    @abstractmethod
    def get_device(self, tensor):
        """Get the device of the tensor"""
        ...

    @abstractmethod
    def to_device(self, tensor, device):
        """Move the tensor to a specified device"""
        ...

    @abstractmethod
    def boolean_ones_like(self, tensor):
        """Create a boolean ones tensor with the same shape as the input tensor"""
        ...

    @abstractmethod
    def apply_mask(self, tensor, mask, value):
        """Fill the elements of the tensor where the mask is True with the specified value"""
        ...
