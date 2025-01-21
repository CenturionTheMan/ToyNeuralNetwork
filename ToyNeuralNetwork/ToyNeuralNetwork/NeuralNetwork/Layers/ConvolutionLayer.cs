using NeuralNetworkLibrary.Math;
using NeuralNetworkLibrary.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Linq;

namespace NeuralNetworkLibrary.NeuralNetwork;

internal class ConvolutionLayer : ILayer
{
    #region PARAMS

    LayerType ILayer.LayerType => LayerType.Convolution;

    private const float maxNorm = 0.5f;

    private int depth;
    private int kernelSize;
    private int stride;

    private ActivationFunction activationFunction;

    private Matrix[,] kernels;
    private float[] biases;

    private Matrix[,] changeForKernels;
    private float[] changeForBiases;

    private int inputDepth;
    private int inputWidth;
    private int inputHeight;

    private int outputColumns;
    private int outputRows;

    #endregion PARAMS

    #region CTOR

    /// <summary>
    /// Initializes a new instance of the <see cref="ConvolutionLayer"/> class.
    /// </summary>
    /// <param name="inputShape">
    /// Shape of the input tensor (depth, height, width)
    /// </param>
    /// <param name="kernelSize">
    /// Size of the kernel (kernelSize x kernelSize)
    /// </param>
    /// <param name="kernelsDepth">
    /// Depth of the kernels (number of kernels)
    /// </param>
    /// <param name="stride">
    /// Stride of the convolution operation. Currently only stride = 1 is supported. Other values are not implemented yet.
    /// </param>
    /// <param name="activationFunction">
    /// Activation function to be used after the convolution operation.
    /// </param>
    /// <exception cref="NotImplementedException">
    /// Exception thrown when stride != 1. Currently only stride = 1 is supported.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Exception thrown when stride is less than 1.
    /// </exception>
    internal ConvolutionLayer((int inputDepth, int inputHeight, int inputWidth) inputShape, int kernelSize, int kernelsDepth, int stride, ActivationFunction activationFunction)
    {
        if (stride != 1)
        {
            throw new NotImplementedException("Stride != 1 is not implemented yet");
        }

        if (stride < 1)
        {
            throw new ArgumentException("Stride must be greater than 0");
        }

        this.depth = kernelsDepth;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.activationFunction = activationFunction;

        this.inputDepth = inputShape.inputDepth;
        this.inputWidth = inputShape.inputWidth;
        this.inputHeight = inputShape.inputHeight;

        kernels = new Matrix[depth, inputShape.inputDepth];
        biases = new float[depth];

        changeForKernels = new Matrix[depth, inputShape.inputDepth];
        changeForBiases = new float[depth];

        (outputRows, outputColumns) = MatrixExtender.GetSizeAfterConvolution((inputHeight, inputWidth), (kernelSize, kernelSize), stride);

        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < this.inputDepth; j++)
            {
                kernels[i, j] = new Matrix(kernelSize, kernelSize);
                switch (activationFunction)
                {
                    case ActivationFunction.ReLU:
                        kernels[i, j].InitializeHe();
                        break;

                    case ActivationFunction.Sigmoid:
                        kernels[i, j].InitializeXavier();
                        break;

                    case ActivationFunction.Softmax:
                        kernels[i, j].InitializeXavier();
                        break;

                    default:
                        throw new NotImplementedException();
                }
                changeForKernels[i, j] = new Matrix(kernelSize, kernelSize);
            }
        }
    }


    /// <summary>
    /// Initializes a new instance of the <see cref="ConvolutionLayer"/> class. 
    /// Used for loading the layer from the file.
    /// </summary>
    /// <param name="layerHead">
    /// Head of the layer containing the layer's parameters.
    /// </param>
    /// <param name="layerData">
    /// Data of the layer containing the layer's weights and biases.
    /// </param>
    /// <returns>
    /// Loaded layer or null if the layer could not be loaded.
    /// </returns>
    internal static ILayer? LoadLayerData(XElement layerHead, XElement layerData)
    {
        string? inputShapeStr = layerHead.Element("inputShape")?.Value;
        string? depthStr = layerHead.Element("depth")?.Value;
        string? kernelSizeStr = layerHead.Element("kernelSize")?.Value;
        string? strideStr = layerHead.Element("stride")?.Value;
        string? activationFunctionStr = layerHead.Element("activationFunction")?.Value;

        if (inputShapeStr == null || depthStr == null || kernelSizeStr == null || strideStr == null || activationFunctionStr == null)
            return null;

        if (!int.TryParse(depthStr, out int depth) || !int.TryParse(kernelSizeStr, out int kernelSize) || !int.TryParse(strideStr, out int stride) || !Enum.TryParse<ActivationFunction>(activationFunctionStr, out ActivationFunction activationFunction))
            return null;

        string[] inputShape = inputShapeStr.Split(' ');
        if (inputShape.Length != 3 || !int.TryParse(inputShape[0], out int inputDepth) || !int.TryParse(inputShape[1], out int inputHeight) || !int.TryParse(inputShape[2], out int inputWidth))
            return null;

        ConvolutionLayer layer = new ConvolutionLayer((inputDepth, inputHeight, inputWidth), kernelSize, depth, stride, activationFunction);

        if (layerData.Element("Biases") == null || layerData.Element("Kernels") == null)
            return null;

        foreach (var kernel in layerData.Element("Kernels")!.Elements("Kernel"))
        {
            string? indexStr = kernel.Attribute("Index")?.Value;
            string? kernelStr = kernel.Value;
            if (indexStr == null || kernelStr == null)
                return null;

            string[] index = indexStr.Split(' ');
            if (index.Length != 2 || !int.TryParse(index[0], out int i) || !int.TryParse(index[1], out int j))
                return null;

            if (!Matrix.TryParse(kernelStr, out Matrix kernelMatrix))
                return null;

            layer.kernels[i, j] = kernelMatrix;
        }

        foreach (var bias in layerData.Element("Biases")!.Elements("Bias"))
        {
            string? indexStr = bias.Attribute("Index")?.Value;
            string? biasStr = bias.Value;
            if (indexStr == null || biasStr == null)
                return null;

            if (!int.TryParse(indexStr, out int i) || !float.TryParse(biasStr, out float biasVal))
                return null;

            layer.biases[i] = biasVal;
        }

        return layer;
    }

    #endregion CTOR

    #region METHODS

    /// <summary>
    /// Forward pass of the convolutional layer.
    /// </summary>
    /// <param name="inputs">
    /// Input tensor to the layer.
    /// </param>
    /// <returns>
    /// Tuple containing the activated output and the output before activation.
    /// </returns>
    (Matrix[] output, Matrix[] otherOutput) ILayer.Forward(Matrix[] inputs)
    {
        // activated output
        Matrix[] A = new Matrix[depth];
        // output before activation
        Matrix[] Z = new Matrix[depth];

        for (int i = 0; i < depth; i++)
        {
            A[i] = new Matrix(outputRows, outputColumns);
            Z[i] = new Matrix(outputRows, outputColumns);
        }

        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                var single = inputs[j].CrossCorrelationValid(kernels[i, j], stride: this.stride);
                Z[i] = Z[i].ElementWiseAdd(single);
            }
            Z[i] = Z[i] + biases[i];

            A[i] = Z[i].ApplyActivationFunction(activationFunction);
        }

        return (A, Z);
    }

    /// <summary>
    /// Backward pass of the convolutional layer.
    /// </summary>
    /// <param name="dAin">
    /// Error gradient from the next layer.
    /// </param>
    /// <param name="layerInputFromForward">
    /// Output from the forward pass of the layer.
    /// </param>
    /// <param name="layerOutputBeforeActivation">
    /// Not activated output from the forward pass of the layer.
    /// </param>
    /// <param name="learningRate">
    /// Learning rate of the neural network.
    /// </param>
    /// <returns>
    /// Error gradient to be passed to the previous layer.
    /// </returns>
    Matrix[] ILayer.Backward(Matrix[] dAin, Matrix[] layerInputFromForward, Matrix[] layerOutputBeforeActivation, float learningRate)
    {
        Matrix[] dA = new Matrix[inputDepth];
        for (int i = 0; i < inputDepth; i++)
        {
            dA[i] = new Matrix(inputHeight, inputWidth);
        }

        Matrix[] dZ = new Matrix[layerOutputBeforeActivation.Length];
        for (int i = 0; i < layerOutputBeforeActivation.Length; i++)
        {
            dZ[i] = Matrix.ElementWiseMultiplyMatrices(dAin[i], layerOutputBeforeActivation[i].DerivativeActivationFunction(activationFunction));
        }

        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < inputDepth; j++)
            {
                Matrix kernelGradient = layerInputFromForward[j].CrossCorrelationValid(dZ[i], stride: this.stride);
                kernelGradient = kernelGradient * learningRate;
                changeForKernels[i, j] = changeForKernels[i, j].ElementWiseAdd(kernelGradient);

                var dASingle = dZ[i].ConvolutionFull(kernels[i, j], stride: this.stride);
                dA[j] = dA[j].ElementWiseAdd(dASingle);
            }

            changeForBiases[i] = changeForBiases[i] + (dZ[i].Sum() * learningRate);
        }

        return dA;
    }

    /// <summary>
    /// Updates the weights and biases of the layer.
    /// </summary>
    /// <param name="batchSize">
    /// size of the batch used for training.
    /// </param>
    void ILayer.UpdateWeightsAndBiases(int batchSize)
    {
        float multiplier = 1.0f / batchSize;

        for (int i = 0; i < depth; i++)
        {
            for (int j = 0; j < kernels.GetLength(1); j++)
            {
                var change = changeForKernels[i, j] * multiplier;
                float clipCoefficient = maxNorm / (change.GetNorm() + float.Epsilon);
                if (clipCoefficient < 1)
                    change = change * clipCoefficient;

                kernels[i, j] = kernels[i, j].ElementWiseAdd(change);
                changeForKernels[i, j] = new Matrix(kernelSize, kernelSize);
            }
            biases[i] += changeForBiases[i] * multiplier;
            changeForBiases[i] = 0;
        }
    }

    #endregion METHODS

    #region SAVE

    void ILayer.SaveLayerDescription(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerHead");
        doc.WriteAttributeString("LayerType", $"{LayerType.Convolution.ToString()}");
        doc.WriteElementString("inputShape", $"{inputDepth} {inputHeight} {inputWidth}");
        doc.WriteElementString("depth", depth.ToString());
        doc.WriteElementString("kernelSize", kernelSize.ToString());
        doc.WriteElementString("stride", stride.ToString());
        doc.WriteElementString("activationFunction", activationFunction.ToString());
        doc.WriteEndElement();
    }

    void ILayer.SaveLayerData(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerData");

        doc.WriteStartElement("Kernels");
        for (int i = 0; i < kernels.GetLength(0); i++)
        {
            for (int j = 0; j < kernels.GetLength(1); j++)
            {
                doc.WriteStartElement("Kernel");
                doc.WriteAttributeString("Index", $"{i} {j}");
                doc.WriteString(kernels[i, j].ToFileString());
                doc.WriteEndElement();
            }
        }
        doc.WriteEndElement();

        doc.WriteStartElement("Biases");
        for (int i = 0; i < biases.Length; i++)
        {
            doc.WriteStartElement("Bias");
            doc.WriteAttributeString("Index", $"{i}");
            doc.WriteString(biases[i].ToString());
            doc.WriteEndElement();
        }
        doc.WriteEndElement();

        doc.WriteEndElement();
    }

    #endregion SAVE
}