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

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
