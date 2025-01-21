using NeuralNetworkLibrary.Math;
using NeuralNetworkLibrary.Utils;
using System.Xml;
using System.Xml.Linq;

namespace NeuralNetworkLibrary.NeuralNetwork;


internal class DropoutLayer : ILayer
{
    #region PARAMS

    LayerType ILayer.LayerType => LayerType.Dropout;

    private static readonly Random random = new Random();
    private (int inputHeight, int inputWidth) inputShape;
    internal float dropoutRate;
    private Matrix mask;

    #endregion PARAMS

    #region CTOR

    /// <summary>
    /// Initializes a new instance of the <see cref="DropoutLayer"/> class.
    /// </summary>
    /// <param name="inputShape">
    /// Shape of the input data.
    /// </param>
    /// <param name="dropoutRate">
    /// Dropout rate. Must be in range [0, 0.9].
    /// </param>
    /// <exception cref="ArgumentException">
    /// Exception thrown when dropout rate is not in range [0, 0.9].
    /// </exception>
    internal DropoutLayer((int inputHeight, int inputWidth) inputShape, float dropoutRate)
    {
        if(dropoutRate < 0 || dropoutRate > 0.9f)
        {
            throw new ArgumentException("Dropout rate must be in range [0, 0.9]");
        }
        this.inputShape = inputShape;
        this.dropoutRate = dropoutRate;
        this.mask = GenerateMask();
    }

    /// <summary>
    /// Load layer data from XML.
    /// </summary>
    /// <param name="layerHead">
    /// Head of the layer data.
    /// </param>
    /// <param name="layerData">
    /// Not used
    /// </param>
    /// <returns></returns>
    internal static DropoutLayer? LoadLayerData(XElement layerHead, XElement layerData)
    {
        string? inputHeightStr = layerHead.Element("InputHeight")?.Value;
        string? inputWidthStr = layerHead.Element("InputWidth")?.Value;
        string? dropoutRateStr = layerHead.Element("DropoutRate")?.Value;

        if (inputHeightStr == null || inputWidthStr == null || dropoutRateStr == null)
        {
            return null;
        }

        if (!int.TryParse(inputHeightStr, out int inputHeight) || !int.TryParse(inputWidthStr, out int inputWidth) || !float.TryParse(dropoutRateStr, out float dropoutRate))
        {
            return null;
        }

        return new DropoutLayer((inputHeight, inputWidth), dropoutRate);
    }

    #endregion CTOR

    #region METHODS

    /// <summary>
    /// Generates mask for dropout layer.
    /// </summary>
    /// <returns>
    /// Returns random mask for dropout layer.
    /// </returns>
    private Matrix GenerateMask()
    {
        var mask = new Matrix(inputShape.inputHeight, inputShape.inputWidth);
        int amountToDrop = (int)(dropoutRate * mask.RowsAmount * mask.ColumnsAmount);

        for (int i = 0; i < mask.RowsAmount; i++)
        {
            for (int j = 0; j < mask.ColumnsAmount; j++)
            {
                mask[i, j] = 1f / (1 - dropoutRate);
            }
        }

        HashSet<(int, int)> droppedIndexes = new HashSet<(int, int)>();
        for (int i = 0; i < amountToDrop; i++)
        {
            int row = -1;
            int column = -1;
            while (row == -1 || droppedIndexes.Contains((row, column)))
            {
                row = random.Next(0, mask.RowsAmount);
                column = random.Next(0, mask.ColumnsAmount);
            }
            droppedIndexes.Add((row, column));        
            mask[row, column] = 0;
        }        

        return mask;
    }

    /// <summary>
    /// Backward pass for dropout layer.
    /// </summary>
    /// <param name="prevOutput"></param>
    /// <param name="prevLayerOutputOther"></param>
    /// <param name="currentLayerOutputOther"></param>
    /// <param name="learningRate"></param>
    /// <returns></returns>
    Matrix[] ILayer.Backward(Matrix[] prevOutput, Matrix[] prevLayerOutputOther, Matrix[] currentLayerOutputOther, float learningRate)
    {
        var tmp = new Matrix[prevOutput.Length];
        for (int i = 0; i < prevOutput.Length; i++)
        {
            tmp[i] = Matrix.ElementWiseMultiplyMatrices(prevOutput[i], mask);
        }
        return tmp;
    }


    /// <summary>
    /// Forward pass for dropout layer.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    (Matrix[] output, Matrix[] otherOutput) ILayer.Forward(Matrix[] inputs)
    {
        var tmp = new Matrix[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            tmp[i] = Matrix.ElementWiseMultiplyMatrices(inputs[i], mask);
        }
        return (tmp, tmp);
    }

    /// <summary>
    /// Updates dropout mask.
    /// </summary>
    /// <param name="batchSize"></param>
    void ILayer.UpdateWeightsAndBiases(int batchSize)
    {
        this.mask = GenerateMask();
    }

    #endregion METHODS

    #region SAVE

    void ILayer.SaveLayerData(XmlTextWriter doc)
    {
       doc.WriteStartElement("LayerData");
        doc.WriteEndElement();
    }

    void ILayer.SaveLayerDescription(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerHead");
        doc.WriteAttributeString("LayerType", LayerType.Dropout.ToString());
        doc.WriteElementString("InputHeight", inputShape.inputHeight.ToString());
        doc.WriteElementString("InputWidth", inputShape.inputWidth.ToString());
        doc.WriteElementString("DropoutRate", dropoutRate.ToString());
        doc.WriteEndElement();
    }

    #endregion SAVE
    
}