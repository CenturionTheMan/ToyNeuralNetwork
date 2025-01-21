using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyNeuralNetwork.Utils;

public enum ActivationFunction
{
    ReLU,
    Sigmoid,
    Softmax,
}

public enum LayerType
{
    Convolution,
    Pooling,
    FullyConnected,
    Dropout,
    Reshape
}