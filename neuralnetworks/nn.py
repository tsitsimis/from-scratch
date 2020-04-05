import numpy as np

from . import utils
from .activations import relu


class Linear:
    """
    Layer performing a linear mapping of the input and applying an activation function

    Methods
    -------
    init_weights()
        Initializes layer weight matrix with random (uniform [0,1]) values

    z(X)
        Calculates the linear mapping of input X when multiplied with weigh matrix

    forward(X)
        Takes the result of `z(X)` and applies the layer's activation function

    backward(accum_grad, layer_input)
        Calculates the weight matrix gradient given the accumulated gradient of
        next layer and updates the accumulated gradient
    """

    def __init__(self, nodes: int, input_size: int = None, activation: callable = relu):
        self.nodes = nodes
        self.input_size = input_size
        self.activation = activation

        self.size = self.nodes
        self.weights = None

        self.accum_grad = None
        self.grad_w = None

    def init_weights(self):
        """
        Initializes layer weight matrix with random (uniform [0,1]) values
        """

        self.weights = np.random.random((self.input_size, self.size)).T

    def z(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the linear mapping of input X when multiplied with weigh matrix

        Parameters
        ----------
        X : numpy.ndarray
            2D input array
        """

        return X.dot(self.weights.T)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Takes the result of `z(X)` and applies the layer's activation function

        Parameters
        ----------
        X : numpy.ndarray
            2D input array
        """

        return self.activation(self.z(X))

    def backward(self, accum_grad: np.ndarray, layer_input: np.ndarray):
        """
        Calculates the weight matrix gradient given the accumulated gradient of
        next layer and updates the accumulated gradient

        Parameters
        ----------
        accum_grad : numpy.ndarray
            Accumulated gradient of next layer. This is basically the gradient of
            the cost function w.r.t. the output of the next layer

        layer_input : numpy.ndarray
            During training this is the result of the forward pass of the network
            until the previous layer
        """

        self.grad_w = (accum_grad * self.activation(self.z(layer_input), derivative=True)).T.dot(layer_input)
        self.accum_grad = (accum_grad * self.activation(self.z(layer_input), derivative=True)).dot(self.weights)

    def __repr__(self):
        if self.size == 2:
            edges = "\  /"
            return f"{'o ' * self.size}\n{'  ' * (self.nodes//2-1)}{edges}"
        elif self.size == 1:
            return " o \n | "
        else:
            edges = "\ | /"

        return f"{'o ' * self.size}\n{'  ' * (self.size//2-2) + (' ' if self.size % 2 == 1 else '')}{edges}"


class NN:
    """
    Neural Network comprising of stacked linear layers`

    Methods
    -------
    forward(X, layer=None)
        Forward pass of the network until a specific layer, or until the last
        if not specified. Iteratively calculates the output of each layer and
        passes the result to the next

    layer_sizes()
        Returns a list with node count of each layer

    init_weights()
        Calls the `init_weights()` method of each layer
    """

    def __init__(self, layers: Linear = None):
        if layers is None:
            layers = []
        self.layers = layers

        self.n_layers = len(self.layers)
        self.weights = None

        self.init_weights()

    def forward(self, X: np.ndarray, layer: int = None) -> np.ndarray:
        """
        Forward pass of the network until a specific layer, or until the last
        if not specified. Iteratively calculates the output of each layer and
        passes the result to the next

        Parameters
        ----------
        X : numpy.ndarray
            2D data array containing input samples

        layer : int
            Index of layer of network until which to make the forward pass
            If None, the forward pass reaches the last layer
        """

        if layer is None:
            layer = self.n_layers - 1

        output = utils.augmented(X)
        for i in range(layer + 1):
            output = self.layers[i].forward(output)

        return output

    def layer_sizes(self) -> list:
        """
        Returns a list with node count of each layer
        """

        return list(map(lambda x: x.size, self.layers))

    def init_weights(self):
        """
        Calls the `init_weights()` method of each layer

        Raises
        ------
        ValueError
            The first layer needs the dimension of the input data

        """

        if self.layers[0].input_size is None:
            raise ValueError("Please specify input dimension in constructor if first layer")
        self.layers[0].init_weights()

        for i in range(1, self.n_layers):
            self.layers[i].input_size = self.layers[i - 1].size
            self.layers[i].init_weights()

    def __repr__(self):
        """
        Draws the network using just ascii characters
        """

        print("Fead-forward Neural Network")
        print("---------------------------")
        print(f"{self.n_layers} layers: {self.layer_sizes()}\n")
        if self.n_layers == 0:
            return ""

        newline = "\n"
        max_layer_size = max(self.layer_sizes())
        print(f"{'  ' * (max_layer_size//2-2)} Inputs")
        for layer in self.layers:
            print(f"{'  ' * (max_layer_size//2 - layer.nodes//2-1)} {str(layer).split(newline)[1]}")
            print(f"{'  ' * (max_layer_size//2 - layer.nodes//2-1)} {str(layer).split(newline)[0]}")
        print(f"{'  ' * (max_layer_size//2-2)} Outputs")
        return ""
