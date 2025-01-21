using NeuralNetworkLibrary.Math;
using NeuralNetworkLibrary.Utils;
using System.Xml;

namespace NeuralNetworkLibrary.NeuralNetwork;

internal interface ILayer
{
    internal LayerType LayerType { get; }

    /// <summary>
    /// Forward pass for the layer
    /// </summary>
    /// <param name="inputs">Data from previous layer</param>
    /// <returns>
    /// output of the layer and other output (max indices in pooling layer, not activated output in
    /// convolution layer)
    /// </returns>
    internal (Matrix[] output, Matrix[] otherOutput) Forward(Matrix[] inputs);

    /// <summary>
    /// Backward pass for the layer
    /// </summary>
    /// <param name="prevOutput">deltas propagated from nextLayer (previous in chain of backward prop)</param>
    /// <param name="currentLayerOutputOther">
    /// output of the current layer. In convolution layer it is
    /// not activated output in pooling layer is ts max indices indexes
    /// </param>
    /// <param name="prevLayerOutputOther">
    /// output of the previous layer (in backprop chain), not activated.
    /// </param>
    internal Matrix[] Backward(Matrix[] prevOutput, Matrix[] prevLayerOutputOther, Matrix[] currentLayerOutputOther, float learningRate);

    /// <summary>
    /// Update weights and biases of the layer.
    /// Values are updated based on the averege of deltas calculated in the backward pass.
    /// </summary>
    /// <param name="batchSize">batch size used in learning</param>
    internal void UpdateWeightsAndBiases(int batchSize);

    /// <summary>
    /// Method to save layer description to xml (header)
    /// </summary>
    /// <param name="doc"></param>
    internal void SaveLayerDescription(XmlTextWriter doc);

    /// <summary>
    /// Method to save layer data to xml (weights and biases)
    /// </summary>
    /// <param name="doc"></param>
    internal void SaveLayerData(XmlTextWriter doc);
}