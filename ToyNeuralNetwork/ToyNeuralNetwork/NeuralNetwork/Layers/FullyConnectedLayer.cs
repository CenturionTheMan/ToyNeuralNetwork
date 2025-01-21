using ToyNeuralNetwork.Math;
using ToyNeuralNetwork.Utils;
using System.Xml;
using System.Xml.Linq;

namespace ToyNeuralNetwork.NeuralNetwork;

internal class FullyConnectedLayer : ILayer
{
    #region PARAMS

    LayerType ILayer.LayerType => LayerType.FullyConnected;

    private const float maxNorm = 0.5f;

    private ActivationFunction activationFunction;
    private int layerSize;

    private Matrix weights;
    private Matrix biases;

    private Matrix weightsGradientSum;
    private Matrix biasesGradientSum;

    #endregion PARAMS

    #region CTOR

    /// <summary>
    /// Initializes a new instance of the <see cref="FullyConnectedLayer"/> class.
    /// </summary>
    /// <param name="previousLayerSize">
    /// Size of the previous layer.
    /// </param>
    /// <param name="layerSize">
    /// Current layer size.
    /// </param>
    /// <param name="activationFunction">
    /// Function used for activation.
    /// </param>
    /// <exception cref="NotImplementedException">
    /// Exception thrown when activation function is not implemented.
    /// </exception>
    internal FullyConnectedLayer(int previousLayerSize, int layerSize, ActivationFunction activationFunction)
    {
        this.weights = new Matrix(layerSize, previousLayerSize);
        this.biases = new Matrix(layerSize, 1);

        switch (activationFunction)
        {
            case ActivationFunction.ReLU:
                this.weights.InitializeHe();
                break;

            case ActivationFunction.Sigmoid:
                this.weights.InitializeXavier();
                break;

            case ActivationFunction.Softmax:
                this.weights.InitializeXavier();
                break;

            default:
                throw new NotImplementedException();
        }

        this.weightsGradientSum = new Matrix(layerSize, previousLayerSize);
        this.biasesGradientSum = new Matrix(layerSize, 1);

        this.activationFunction = activationFunction;
        this.layerSize = layerSize;
    }

    /// <summary>
    /// Load layer data from XML.
    /// </summary>
    /// <param name="layerHead"></param>
    /// <param name="layerData"></param>
    /// <returns></returns>
    internal static ILayer? LoadLayerData(XElement layerHead, XElement layerData)
    {
        string? previousLayerSizeStr = layerHead.Element("previousLayerSize")?.Value;
        string? layerSizeStr = layerHead.Element("layerSize")?.Value;
        string? activationFunctionStr = layerHead.Element("activationFunction")?.Value;

        string? weightsStr = layerData.Element("Weights")?.Value;
        string? biasesStr = layerData.Element("Biases")?.Value;

        if (previousLayerSizeStr == null || layerSizeStr == null || activationFunctionStr == null || weightsStr == null || biasesStr == null)
            return null;

        if (!int.TryParse(previousLayerSizeStr, out int previousLayerSize) || !int.TryParse(layerSizeStr, out int layerSize) || !Enum.TryParse<ActivationFunction>(activationFunctionStr, out ActivationFunction activationFunction))
            return null;

        if (!Matrix.TryParse(weightsStr, out Matrix weights) || !Matrix.TryParse(biasesStr, out Matrix biases))
            return null;

        FullyConnectedLayer layer = new FullyConnectedLayer(previousLayerSize, layerSize, activationFunction);
        layer.weights = weights;
        layer.biases = biases;
        return layer;
    }

    #endregion CTOR

    #region METHODS

    /// <summary>
    /// Forward pass for fully connected layer.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException">
    /// Exception thrown when fully connected layer has more than one input.
    /// </exception>
    /// <exception cref="NotImplementedException">
    /// Exception thrown when activation function is not implemented.
    /// </exception>
    (Matrix[] output, Matrix[] otherOutput) ILayer.Forward(Matrix[] input)
    {
        if (input.Length != 1)
            throw new ArgumentException("Fully connected layer can only have one input");

        Matrix currentLayer = input[0];

        Matrix multipliedByWeightsLayer = Matrix.DotProductMatrices(weights, currentLayer);
        Matrix layerWithAddedBiases = multipliedByWeightsLayer.ElementWiseAdd(biases);

        Matrix activatedLayer = activationFunction switch
        {
            ActivationFunction.ReLU => ActivationFunctionsHandler.ReLU(layerWithAddedBiases),
            ActivationFunction.Sigmoid => ActivationFunctionsHandler.Sigmoid(layerWithAddedBiases),
            ActivationFunction.Softmax => ActivationFunctionsHandler.Softmax(layerWithAddedBiases),
            _ => throw new NotImplementedException()
        };

        return ([activatedLayer], [layerWithAddedBiases]);
    }

    /// <summary>
    /// Backward pass for fully connected layer.
    /// </summary>
    Matrix[] ILayer.Backward(Matrix[] errorMatrix, Matrix[] prevLayerOutputActivated, Matrix[] thisLayerOutputBeforeActivation, float learningRate)
    {
        if (errorMatrix.Length != 1)
            throw new ArgumentException("Fully connected layer can only have one input");

        Matrix activationDerivativeLayer = activationFunction switch
        {
            ActivationFunction.ReLU => ActivationFunctionsHandler.DerivativeReLU(thisLayerOutputBeforeActivation[0]),
            ActivationFunction.Sigmoid => ActivationFunctionsHandler.DerivativeSigmoid(thisLayerOutputBeforeActivation[0]),
            ActivationFunction.Softmax => ActivationFunctionsHandler.DerivativeSoftmax(thisLayerOutputBeforeActivation[0]),
            _ => throw new NotImplementedException()
        };

        Matrix gradientMatrix = activationDerivativeLayer.ElementWiseMultiply(errorMatrix[0]).ApplyFunction(x => x * learningRate);
        Matrix deltaWeightsMatrix = Matrix.DotProductMatrices(gradientMatrix, prevLayerOutputActivated[0].Transpose());

        weightsGradientSum = weightsGradientSum.ElementWiseAdd(deltaWeightsMatrix);
        biasesGradientSum = biasesGradientSum.ElementWiseAdd(gradientMatrix);

        return [Matrix.DotProductMatrices(weights.Transpose(), errorMatrix[0])];
    }

    /// <summary>
    /// Update weights and biases.
    /// </summary>
    /// <param name="batchSize">
    /// Size of the batch.
    /// </param>
    void ILayer.UpdateWeightsAndBiases(int batchSize)
    {
        float multiplier = 1.0f / batchSize;
        var changeForWeights = weightsGradientSum * multiplier;

        float clipCoefficient = maxNorm / (changeForWeights.GetNorm() + float.Epsilon);
        if (clipCoefficient < 1)
            changeForWeights = changeForWeights * clipCoefficient;

        weights = weights.ElementWiseAdd(changeForWeights);
        biases = biases.ElementWiseAdd(biasesGradientSum.ApplyFunction(x => x * multiplier));

        weightsGradientSum = new Matrix(layerSize, weights.ColumnsAmount);
        biasesGradientSum = new Matrix(layerSize, 1);
    }

    #endregion METHODS

    #region SAVE

    void ILayer.SaveLayerDescription(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerHead");
        doc.WriteAttributeString("LayerType", $"{LayerType.FullyConnected.ToString()}");
        doc.WriteElementString("previousLayerSize", weights.ColumnsAmount.ToString());
        doc.WriteElementString("layerSize", layerSize.ToString());
        doc.WriteElementString("activationFunction", activationFunction.ToString());
        doc.WriteEndElement();
    }

    void ILayer.SaveLayerData(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerData");

        doc.WriteElementString("Weights", weights.ToFileString());
        doc.WriteElementString("Biases", biases.ToFileString());

        doc.WriteEndElement();
    }

    #endregion SAVE
}