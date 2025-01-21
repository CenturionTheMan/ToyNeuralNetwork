using NeuralNetworkLibrary.Math;
using NeuralNetworkLibrary.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Linq;

namespace NeuralNetworkLibrary.NeuralNetwork;

internal class PoolingLayer : ILayer
{
    #region PARAMS

    LayerType ILayer.LayerType => LayerType.Pooling;

    private int poolSize;
    private int stride;

    #endregion PARAMS

    #region CTOR

    /// <summary>
    /// Initializes a new instance of the <see cref="PoolingLayer"/> class.
    /// </summary>
    /// <param name="poolSize">
    /// Size of the pooling window.
    /// </param>
    /// <param name="stride">
    /// Stride of the pooling window.
    /// </param>
    internal PoolingLayer(int poolSize, int stride)
    {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    /// <summary>
    /// Load layer data from XML.
    /// </summary>
    /// <param name="layerHead"></param>
    /// <param name="layerData"></param>
    /// <returns></returns>
    internal static ILayer? LoadLayerData(XElement layerHead, XElement layerData)
    {
        string? poolSizeStr = layerHead.Element("PoolSize")?.Value;
        string? strideStr = layerHead.Element("Stride")?.Value;

        if (poolSizeStr == null || strideStr == null)
            return null;

        if (!int.TryParse(poolSizeStr, out int poolSize) || !int.TryParse(strideStr, out int stride))
            return null;

        return new PoolingLayer(poolSize, stride);
    }

    #endregion CTOR

    #region METHODS

    /// <summary>
    /// Does nothing.
    /// </summary>
    /// <param name="batchSize"></param>
    void ILayer.UpdateWeightsAndBiases(int batchSize)
    {
        //nothing to do here
    }

    /// <summary>
    /// Forward pass of the pooling layer.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    (Matrix[] output, Matrix[] otherOutput) ILayer.Forward(Matrix[] inputs)
    {
        // this.previousLayerOutputs = inputs;

        Matrix[] result = new Matrix[inputs.Length];
        Matrix[] maxIndexMap = new Matrix[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            (result[i], maxIndexMap[i]) = MatrixExtender.MaxPooling(inputs[i], poolSize, stride);
        }
        return (result, maxIndexMap);
    }

    /// <summary>
    /// Backward pass of the pooling layer.
    /// </summary>
    /// <param name="deltas"></param>
    /// <param name="prevLayerSize"></param>
    /// <param name="maxIndexMap"></param>
    /// <param name="learningRate"></param>
    /// <returns></returns>
    Matrix[] ILayer.Backward(Matrix[] deltas, Matrix[] prevLayerSize, Matrix[] maxIndexMap, float learningRate)
    {
        Matrix[] result = new Matrix[deltas.Length];
        for (int i = 0; i < deltas.Length; i++)
        {
            result[i] = new Matrix(prevLayerSize[i].RowsAmount, prevLayerSize[i].ColumnsAmount);

            for (int j = 0; j < deltas[i].RowsAmount; j++)
            {
                for (int k = 0; k < deltas[i].ColumnsAmount; k++)
                {
                    int maxIndex = (int)maxIndexMap[i][j, k];

                    var dim = MatrixHelpers.GetRowAndColumn(maxIndex, prevLayerSize[i].ColumnsAmount);
                    result[i][dim.row, dim.column] += deltas[i][j, k];
                }
            }
        }
        return result;
    }

    #endregion METHODS

    #region SAVE

    void ILayer.SaveLayerDescription(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerHead");
        doc.WriteAttributeString("LayerType", LayerType.Pooling.ToString());
        doc.WriteElementString("PoolSize", poolSize.ToString());
        doc.WriteElementString("Stride", stride.ToString());
        doc.WriteEndElement();
    }

    void ILayer.SaveLayerData(XmlTextWriter doc)
    {
        doc.WriteStartElement("LayerData");
        doc.WriteEndElement();
    }

    #endregion SAVE
}