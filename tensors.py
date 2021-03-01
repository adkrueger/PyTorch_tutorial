import torch
import numpy as np

# creating a tensor
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# can also be done with numpy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# or, from another tensor
#   (note that new tensors retain properties (shape and datatype) of the argument tensor unless overridden)
x_ones = torch.ones_like(x_data)    # retain properties
print(f'Ones tensor:\n {x_ones}\n')
x_rand = torch.rand_like(x_data, dtype=torch.float)     # override properties
print(f'Random tensor:\n {x_rand}\n')

# shape is a tuple of tensor dimensions; here it will determine the dimensionality of the output tensor
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f'Random Tensor: \n {rand_tensor} \n')
print(f'Ones Tensor: \n {ones_tensor} \n')
print(f'Zeros Tensor: \n {zeros_tensor}\n')

# tensors have attributes like shape, datatype, and the device they're stored on
tensor = torch.rand(3, 4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}\n')

# there's also many tensor operations: https://pytorch.org/docs/stable/torch.html
# for example, moving operations to the GPU:
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# similar operations to numpy
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)

# torch.cat will concatenate a sequence of tensors along a given dimension
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# a few ways to compute the element-wise product:
print(f'tensor.mul(tensor):\n {tensor.mul(tensor)}\n')
print(f'tensor * tensor:\n {tensor * tensor}\n')

# alternatively, matrix multiplication
print(f'tensor.matmul(tensor.T):\n {tensor.matmul(tensor.T)} \n')
print(f'tensor @ tensor.T: \n {tensor @ tensor.T}\n')

# there's also in-place operations that have a _ suffix
# (note that these save memory, but can be problematic with derivatives because we lose history)
print(tensor, '\n')
tensor.add_(5)
print(tensor)

# tensors on CPU and numpy arrays can share underlying memory locations so that changing one will change the other
t = torch.ones(5)
print(f'\nt: {t}')
n = t.numpy()
print(f'n: {n}')
# now that it's bridged, lets see the changes:
t.add_(1)
print(f't: {t}')
print(f'n: {n}')

# numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f't: {t}')
print(f'n: {n}')
