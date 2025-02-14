import pickle
import jax
import jax.numpy as jnp
from dataclasses import asdict
import jax.tree_util as tree
import numpy as np
import torch.onnx
import torch.nn as nn



class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes,
        activation=nn.ReLU(),
        kernel_init="lecun_uniform",
        activate_final=False,
        bias=True,
        layer_norm=False,
        mean_std=None,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.kernel_init = kernel_init
        self.activate_final = activate_final
        self.bias = bias
        self.layer_norm = layer_norm

        # Register mean and std as buffers (non-trainable parameters)
        if mean_std is not None:
            self.register_buffer('mean', mean_std[0].clone().detach())  # Use .clone().detach()
            self.register_buffer('std', mean_std[1].clone().detach())   # Use .clone().detach()
           
        else:
            self.mean = None
            self.std = None

        # Build the MLP block
        self.mlp_block = nn.Sequential()
        for i in range(len(self.layer_sizes) - 1):
            in_features = self.layer_sizes[i]
            out_features = self.layer_sizes[i + 1]

            # Add linear layer
            dense_layer = nn.Linear(in_features, out_features, bias=self.bias)
            self.mlp_block.add_module(f"hidden_{i}", dense_layer)

            # Initialize weights (e.g., Lecun uniform initialization)
            if self.kernel_init == "lecun_uniform":
                nn.init.kaiming_uniform_(dense_layer.weight, mode='fan_in', nonlinearity='relu')

            # Add layer normalization if enabled
            if self.layer_norm and i < len(self.layer_sizes) - 2:  # No layer norm after the last layer
                self.mlp_block.add_module(f"layer_norm_{i}", nn.LayerNorm(out_features))

            # Add activation function, except for the final layer if `activate_final` is False
            if i < len(self.layer_sizes) - 2 or self.activate_final:  # Add activation for all but the last layer
                self.mlp_block.add_module(f"activation_{i}", self.activation)

    def forward(self, inputs):
        # Handle list inputs
        if isinstance(inputs, list):
            inputs = inputs[0]

        # Normalize inputs if mean and std are provided
        if self.mean is not None and self.std is not None:
            inputs = (inputs - self.mean) / self.std

        # Pass through the MLP block
        logits = self.mlp_block(inputs)

        # Split the output into two parts and apply tanh to the first half
        loc, _ = torch.split(logits, logits.size(-1) // 2, dim=-1)
        return torch.tanh(loc)

def make_policy_network(
    observation_size,
    action_size,
    mean_std,
    hidden_layer_sizes=[256, 256],
    activation=nn.ReLU(),
    kernel_init="lecun_uniform",
    layer_norm=False,
):
    layers = [observation_size] + hidden_layer_sizes + [action_size]
    policy_network = MLP(
        layer_sizes= layers,
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        mean_std=mean_std,
    )
    return policy_network



if __name__ == "__main__":


    obs_size = 48
    act_size = 12

    weights_path = "go2_params.pkl"

    with open(weights_path, "rb") as f:
        params_loaded = pickle.load(f)

    # Convert back to JAX arrays if needed
    params_jax = jax.tree.map(jnp.array, params_loaded)
    print("Params successfully loaded")


    mean_std_all = asdict(params_loaded[0])
    weights_bias = params_loaded[1]


    mean_std = (torch.tensor(mean_std_all['mean']['state']), torch.tensor(mean_std_all['std']['state']))

    th_policy_network = make_policy_network(
        observation_size = obs_size,
        action_size = act_size*2,
        mean_std=mean_std,
        hidden_layer_sizes=[512, 256, 128])

    #copy weight to the torch network

    # Assuming th_policy_network is already defined
    values = [(key,value) for key, value in weights_bias["params"].items()]
    j = 0
    for i, layer in enumerate(th_policy_network.mlp_block):
        if isinstance(layer, nn.Linear):  # Check if the layer is a Linear layer
            if (i%2==0):
                transpose_tensor_kernel = torch.tensor(values[j][1]['kernel']).t()
                transpose_tensor_bias = torch.tensor(values[j][1]['bias']).t()
                layer.weight.data = transpose_tensor_kernel
                layer.bias.data = transpose_tensor_bias
                j = j + 1
        
    batch_size = 1
    dummy_input = torch.randn(batch_size, 48)  # For a batch of inputs

    th_policy_network.forward(dummy_input)

    # Define the output ONNX file path
    onnx_file_path = "utils/outputs/go22_policy.onnx"

    # Export the model
    torch.onnx.export(
        th_policy_network,                  # Model to export
        dummy_input,            # Dummy input
        onnx_file_path,         # Output file path
        export_params=True,     # Export model parameters (weights)
        opset_version=11,       # ONNX opset version (e.g., 11 is widely supported)
        do_constant_folding=True,  # Optimize the model by folding constants
        input_names=["state"],  # Input tensor name
        output_names=["actions"],  # Output tensor name
    )

    print(f"Model exported to {onnx_file_path}")