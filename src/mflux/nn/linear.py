import mlx.core as mx


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = mx.random.normal((out_features, in_features), dtype=mx.float32)
        if bias:
            self.bias = mx.zeros((1, out_features), dtype=mx.float32)  # Correct bias shape
        else:
            self.bias = None

    def __call__(self, x):
        x = mx.matmul(x, self.weight.T)  # Use mx.matmul for matrix multiplication
        if self.bias is not None:
            bias = mx.broadcast_to(self.bias, (x.shape[0], self.bias.shape[1]))
            x = x + bias  # Bias addition
        return x
