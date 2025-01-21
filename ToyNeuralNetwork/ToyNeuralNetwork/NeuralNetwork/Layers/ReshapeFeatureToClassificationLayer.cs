using NeuralNetworkLibrary.Math;
using NeuralNetworkLibrary.Utils;
using System.Xml;
using System.Xml.Linq;

namespace NeuralNetworkLibrary.NeuralNetwork;

internal class ReshapeFeatureToClassificationLayer : ILayer
{
    #region PARAMS

    LayerType ILayer.LayerType => LayerType.Reshape;

    private int rowsAmount;
    private int columnsAmount;

    #endregion PARAMS

    #region CTOR

    /// <summary>
    /// Initializes a new instance of the <see cref="ReshapeFeatureToClassificationLayer"/> class.
    /// </summary>
    /// <param name="rowsAmount">
    /// Rows amount of the input data.
    /// </param>
    /// <param name="columnsAmount">
    /// Columns amount of the input data.
    /// </param>
    internal ReshapeFeatureToClassificationLayer(int rowsAmount, int columnsAmount)
    {
        this.rowsAmount = rowsAmount;
        this.columnsAmount = columnsAmount;
    }

    /// <summary>
    /// Load layer data from XML.
    /// </summary>
    /// <param name="layerHead"></param>
    /// <param name="layerData"></param>
    /// <returns></returns>
    internal static ILayer? LoadLayerData(XElement layerHead, XElement layerData)
    {
        string? rowsAmount = layerHead.Element("RowsAmount")?.Value;
        string? columnsAmount = layerHead.Element("ColumnsAmount")?.Value;

        if (rowsAmount == null || columnsAmount == null)
            return null;

        return new ReshapeFeatureToClassificationLayer(int.Parse(rowsAmount), int.Parse(columnsAmount));
    }

    #endregion CTOR

    #region METHODS

    /// <summary>
    /// Forward pass of the layer.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    (Matrix[] output, Matrix[] otherOutput) ILayer.Forward(Matrix[] inputs)
    {
        var flattenedMatrix = MatrixExtender.FlattenMatrix(inputs);
        return ([flattenedMatrix], [flattenedMatrix]);
    }

    /// <summary>
    /// Backward pass of the layer.
    /// </summary>
    /// <param name="prevOutput"></param>
    /// <param name="prevLayerOutputOther"></param>
    /// <param name="currentLayerOutputOther"></param>
    /// <param name="learningRate"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    Matrix[] ILayer.Backward(Matrix[] prevOutput, Matrix[] prevLayerOutputOther, Matrix[] currentLayerOutputOther, float learningRate)
    {
        if (prevOutput.Length != 1)
            throw new ArgumentException("Reshape layer can only have one input");

        Matrix[] errorMatrices = MatrixExtender.UnflattenMatrix(prevOutput[0], rowsAmount, columnsAmount);
        return errorMatrices;
    }

    void ILayer.UpdateWeightsAndBiases(int batchSize)
    {
        //nothing to do here
    }

    #endregion METHODS

    #region SAVE

    void ILayer.SaveLayerDescription(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerHead");
        doc.WriteAttributeString("LayerType", $"{LayerType.Reshape.ToString()}");
        doc.WriteElementString("RowsAmount", rowsAmount.ToString());
        doc.WriteElementString("ColumnsAmount", columnsAmount.ToString());
        doc.WriteEndElement();
    }

    void ILayer.SaveLayerData(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerData");
        doc.WriteEndElement();
    }

    #endregion SAVE
}