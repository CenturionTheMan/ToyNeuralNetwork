using ToyNeuralNetwork.Utils;

namespace ToyNeuralNetwork.NeuralNetwork;

public class LayerTemplate
{   
    internal LayerType LayerType => layerType;
    internal ActivationFunction ActivationFunction => activationFunction;

    internal int LayerSize => layerSize;

    internal int Stride => stride;
    internal int PoolSize => poolSize;
    internal int KernelSize => kernelSize;
    internal int Depth => depth;

    internal float DropoutRate => dropoutRate;


    private LayerType layerType;
    private ActivationFunction activationFunction;

    private int layerSize;

    private int stride;
    private int poolSize;
    private int kernelSize;
    private int depth;

    private float dropoutRate;

    private LayerTemplate()
    {

    }

    /// <summary>
    /// Create a fully connected layer.
    /// </summary>
    /// <param name="layerSize">
    /// Size of the layer.
    /// </param>
    /// <param name="activationFunction">
    /// Activation function of the layer.
    /// </param>
    /// <returns>
    /// Template of the fully connected layer.
    /// </returns>
    public static LayerTemplate CreateFullyConnectedLayer(int layerSize, ActivationFunction activationFunction)
    {
        return new LayerTemplate
        {
            layerType = LayerType.FullyConnected,
            activationFunction = activationFunction,
            layerSize = layerSize
        };
    }

    /// <summary>
    /// Create a max pooling layer.
    /// </summary>
    /// <param name="poolSize">
    /// Size of the pooling window.
    /// </param>
    /// <param name="stride">
    /// Stride of the pooling window.
    /// </param>
    /// <returns>
    /// Template of the max pooling layer.
    /// </returns>
    public static LayerTemplate CreateMaxPoolingLayer(int poolSize, int stride)
    {
        return new LayerTemplate
        {
            layerType = LayerType.Pooling,
            poolSize = poolSize,
            stride = stride
        };
    }


    /// <summary>
    /// Create a convolution layer.
    /// </summary>
    /// <param name="kernelSize">
    /// Size of the kernel (width and height).
    /// </param>
    /// <param name="depth">
    /// Depth of the kernel (amount).
    /// </param>
    /// <param name="activationFunction">
    /// Activation function of the layer.
    /// </param>
    /// <returns>
    /// Template of the convolution layer.
    /// </returns>
    /// <exception cref="NotImplementedException">
    /// Exception thrown when stride is other than 1 (not implemented yet)
    /// </exception>
    public static LayerTemplate CreateConvolutionLayer(int kernelSize, int depth, ActivationFunction activationFunction)
    {
        const int stride = 1;
        if(stride != 1)
            throw new NotImplementedException("Stride != 1 is not implemented yet");

        return new LayerTemplate
        {
            layerType = LayerType.Convolution,
            kernelSize = kernelSize,
            depth = depth,
            stride = stride,
            activationFunction = activationFunction,
        };
    }

    /// <summary>
    /// Create a dropout layer.
    /// </summary>
    /// <param name="dropoutRate">
    /// Rate of the dropout. Must be in range [0, 0.9].
    /// </param>
    /// <returns>
    /// Template of the dropout layer.
    /// </returns>
    public static LayerTemplate CreateDropoutLayer(float dropoutRate)
    {
        if(dropoutRate < 0 || dropoutRate > 0.9f)
            throw new ArgumentException("Dropout rate must be in range [0, 0.9]");

        return new LayerTemplate
        {
            layerType = LayerType.Dropout,
            dropoutRate = dropoutRate,
        };
    }
}