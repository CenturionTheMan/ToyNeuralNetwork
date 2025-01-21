using NeuralNetworkLibrary.Math;
using NeuralNetworkLibrary.Utils;
using System.Xml.Linq;

using static NeuralNetworkLibrary.ImageProcessing.ImageEditor;

namespace NeuralNetworkLibrary.NeuralNetwork;

public class NeuralNetwork
{
    #region PARAMS

    public Action<int, int, float>? OnTrainingIteration; //epoch, sample index, error
    public Action<int, float, float>? OnBatchTrainingIteration; //epoch, epochPercentFinish, error(mean)
    public Action<int, float>? OnEpochTrainingIteration; //epoch, correctness
    public Action? OnTrainingFinished;

    public float LearningRate { get; internal set; }
    public float LastTrainCorrectness { get; internal set; }

    private static Random random = new Random();
    private ILayer[] layers;
    private Dictionary<ILayer, float> layersDropoutRates = new Dictionary<ILayer, float>();

    private int inputDepth;
    private int inputRowsAmount;
    private int inputColumnsAmount;

    #endregion PARAMS

    #region CTORS

    /// <summary>
    /// Creates a new neural network with given layers and learning rate
    /// used for loading neural network from file
    /// </summary>
    /// <param name="inputDepth"> Input depth (channels amount) </param>
    /// <param name="inputRowsAmount"> Row amount (height) </param>
    /// <param name="inputColumnsAmount"> Columns amount (width) </param>
    /// <param name="layers">  </param>
    /// <param name="learningRate"></param>
    /// <param name="lastTrainCorrectness"></param>
    /// <param name="layersDropoutRates"></param>
    private NeuralNetwork(int inputDepth, int inputRowsAmount, int inputColumnsAmount, ILayer[] layers, float learningRate, float lastTrainCorrectness, Dictionary<ILayer, float> layersDropoutRates)
    {
        this.layers = layers;
        this.LearningRate = learningRate;
        this.LastTrainCorrectness = lastTrainCorrectness;

        this.inputDepth = inputDepth;
        this.inputRowsAmount = inputRowsAmount;
        this.inputColumnsAmount = inputColumnsAmount;

        this.layersDropoutRates = layersDropoutRates;
    }

    public NeuralNetwork(int inputSize, LayerTemplate[] layerTemplates) : this(1, inputSize, 1, layerTemplates)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralNetwork"/> class with the specified parameters.
    /// Will throw an exception if the configuration or parameters values invalid.
    /// </summary>
    /// <param name="inputDepth">The depth of the input data.</param>
    /// <param name="inputRowsAmount">The number of rows in the input data.</param>
    /// <param name="inputColumnsAmount">The number of columns in the input data.</param>
    /// <param name="layerTemplates">An array of <see cref="LayerTemplate"/> objects representing the layer configurations.</param>
    public NeuralNetwork(int inputDepth, int inputRowsAmount, int inputColumnsAmount, LayerTemplate[] layerTemplates)
    {
        if(layerTemplates.Length == 0)
            throw new ArgumentException("At least one layer should be provided");

        if (layerTemplates[^1].LayerType != LayerType.FullyConnected)
            throw new ArgumentException("Last layer should be fully connected");

        if (layerTemplates[^1].ActivationFunction != ActivationFunction.Softmax)
            throw new NotImplementedException("Last layer should have softmax activation function");


        this.inputDepth = inputDepth;
        this.inputRowsAmount = inputRowsAmount;
        this.inputColumnsAmount = inputColumnsAmount;

        List<ILayer> layers = new List<ILayer>();
        var currentInput = (inputDepth, inputRowsAmount, inputColumnsAmount);

        for (int i = 0; i < layerTemplates.Length; i++)
        {
            var currentTemplate = layerTemplates[i];
            switch (currentTemplate.LayerType)
            {
                case LayerType.Convolution:
                    if (layers.Count() > 0 && layers.Last().LayerType == LayerType.FullyConnected)
                    {
                        throw new InvalidOperationException("Convolution layer should be after another convolution layer or pooling layer");
                    }

                    var layer = new ConvolutionLayer(currentInput, currentTemplate.KernelSize, currentTemplate.Depth, currentTemplate.Stride, currentTemplate.ActivationFunction);
                    layers.Add(layer);
                    var size = MatrixExtender.GetSizeAfterConvolution((currentInput.inputRowsAmount, currentInput.inputColumnsAmount), (currentTemplate.KernelSize, currentTemplate.KernelSize), currentTemplate.Stride);
                    currentInput = (currentTemplate.Depth, size.outputRows, size.outputColumns);
                    break;

                case LayerType.Pooling:
                    if (layers.Count() > 0 && layers.Last().LayerType == LayerType.FullyConnected)
                    {
                        throw new InvalidOperationException("Pooling layer should be after another convolution layer or pooling layer");
                    }

                    var poolingLayer = new PoolingLayer(currentTemplate.PoolSize, currentTemplate.Stride);
                    layers.Add(poolingLayer);
                    var poolingSize = MatrixExtender.GetSizeAfterPooling((currentInput.inputRowsAmount, currentInput.inputColumnsAmount), currentTemplate.PoolSize, currentTemplate.Stride);
                    currentInput = (currentInput.inputDepth, poolingSize.outputRows, poolingSize.outputColumns);
                    break;

                case LayerType.FullyConnected:
                    if (layers.Count() > 0 && layers.Last().LayerType != LayerType.FullyConnected)
                    {
                        var reshapeLayer = ReshapeInput(true, currentInput.inputRowsAmount, currentInput.inputColumnsAmount);
                        currentInput = (1, currentInput.inputRowsAmount * currentInput.inputColumnsAmount * currentInput.inputDepth, 1);
                        layers.Add(reshapeLayer);
                    }

                    var fullyConnectedLayer = new FullyConnectedLayer(currentInput.inputRowsAmount, currentTemplate.LayerSize, currentTemplate.ActivationFunction);
                    layers.Add(fullyConnectedLayer);
                    currentInput = (1, currentTemplate.LayerSize, 1);
                    break;

                case LayerType.Dropout:
                    if (layers.Count() == 0)
                    {
                        throw new InvalidOperationException("Dropout layer can t be first in sequance.");
                    }
                    var dropoutLayer = new DropoutLayer((currentInput.inputRowsAmount, currentInput.inputColumnsAmount), currentTemplate.DropoutRate);
                    layers.Add(dropoutLayer);
                    layersDropoutRates.Add(dropoutLayer, currentTemplate.DropoutRate);
                    break;

                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        this.layers = layers.ToArray();
    }

    /// <summary>
    /// Create reshape layer based on the input parameters
    /// </summary>
    /// <param name="featureToClassification">
    /// If true will be created layer which reshapes feature layer into classification
    /// </param>
    /// <param name="rows">from layer height</param>
    /// <param name="columns">from layer width</param>
    /// <returns>New instance of <see cref="ReshapeFeatureToClassificationLayer"/>  </returns>
    /// <exception cref="NotImplementedException">Reshaping classification -> feature not yet implemented</exception>
    private ILayer ReshapeInput(bool featureToClassification, int rows, int columns)
    {
        if (featureToClassification)
        {
            var reshape = new ReshapeFeatureToClassificationLayer(rows, columns);
            return reshape;
        }
        else
        {
            throw new NotImplementedException("Reshaping classification -> feature not yet implemented");
        }
    }

    #endregion CTORS

    #region TRAINING

    /// <summary>
    /// Trains the neural network on a new task asynchronously.
    /// </summary>
    /// <param name="data">The training data consisting of input channels and corresponding output.</param>
    /// <param name="learningRate">The learning rate for the training process.</param>
    /// <param name="epochAmount">The number of epochs to train for.</param>
    /// <param name="batchSize">The size of each training batch.</param>
    /// <param name="cancellationToken">The cancellation token to cancel the training process.</param>
    /// <returns>A task representing the asynchronous training process.</returns>
    public Task TrainOnNewTask((Matrix[] inputChannels, Matrix output)[] data, float learningRate, int epochAmount, int batchSize, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Train(data, learningRate, epochAmount, batchSize, cancellationToken), cancellationToken);
    }

    /// <summary>
    /// Trains the neural network using the provided data.
    /// </summary>
    /// <param name="data">The training data, consisting of input channels and corresponding output.</param>
    /// <param name="learningRate">The learning rate for the training process.</param>
    /// <param name="epochAmount">The number of epochs to train for.</param>
    /// <param name="batchSize">The size of each training batch.</param>
    /// <param name="cancellationToken">The cancellation token to stop the training process.</param>
    public void Train((Matrix[] inputChannels, Matrix output)[] data, float learningRate, int epochAmount, int batchSize, CancellationToken cancellationToken = default)
    {
        this.LearningRate = learningRate;

        for (int epoch = 0; epoch < epochAmount; epoch++)
        {
            data = data.OrderBy(x => random.Next()).ToArray();
            int batchBeginIndex = 0;

            while (batchBeginIndex < data.Length)
            {
                var batchSamples = batchBeginIndex + batchSize < data.Length ? data.Skip(batchBeginIndex).Take(batchSize).ToArray() : data[batchBeginIndex..].ToArray();

                float batchErrorSum = 0;

                Parallel.For(0, batchSamples.Length, (i, loopState) =>
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        loopState.Stop();
                        return;
                    }

                    (Matrix prediction, (Matrix[] activated, Matrix[] beforeActivation)[] layersOutputs) = Feedforward(batchSamples[i].inputChannels);
                    prediction = prediction + float.Epsilon;

                    float error = ActivationFunctionsHandler.CalculateCrossEntropyCost(batchSamples[i].output, prediction);
                    batchErrorSum += error;

                    Backpropagation(batchSamples[i].output, prediction, layersOutputs);

                    OnTrainingIteration?.Invoke(epoch + 1, batchBeginIndex + i, error);
                });

                if (cancellationToken.IsCancellationRequested)
                {
                    OnTrainingFinished?.Invoke();
                    return;
                }

                foreach (var layer in layers)
                {
                    layer.UpdateWeightsAndBiases(batchSize);
                }

                float epochPercentFinish = 100 * batchBeginIndex / (float)data.Length;
                OnBatchTrainingIteration?.Invoke(epoch + 1, epochPercentFinish, batchErrorSum / batchSize);

                batchBeginIndex += batchSize;
            }

            float correctness = CalculateCorrectness(data.Take(1000).OrderBy(x => random.Next()).ToArray());
            this.LastTrainCorrectness = correctness;
            OnEpochTrainingIteration?.Invoke(epoch + 1, correctness);
        }

        OnTrainingFinished?.Invoke();
    }

    #endregion TRAINING

    #region INTERACTIONS

    public bool IsConvolutional()
    {
        return layers.Any(x => x.LayerType == LayerType.Convolution || x.LayerType == LayerType.Pooling);
    }

    public (int depth, int rowsAmount, int columnsAmount) GetInputShape()
    {
        return (inputDepth, inputRowsAmount, inputColumnsAmount);
    }

    /// <summary>
    /// Predicts the output based on the input channel.
    /// </summary>
    /// <param name="inputChannel"> Single channel of input </param>
    /// <returns> Prediction </returns>
    public Matrix Predict(Matrix inputChannel)
    {
        return Predict([inputChannel]);
    }

    /// <summary>
    /// Predicts the output based on the input channels.
    /// </summary>
    /// <param name="inputChannels"> Channels of input </param>
    /// <returns> Prediction </returns>
    public Matrix Predict(Matrix[] inputChannels)
    {
        Matrix[] currentInput = inputChannels;

        for (int i = 0; i < layers.Length; i++)
        {
            if (layers[i].LayerType == LayerType.Dropout)
            {
                float func = 1 - layersDropoutRates[layers[i]];
                currentInput = currentInput.Select(x => x * func).ToArray();
            }
            else
            {
                (currentInput, _) = layers[i].Forward(currentInput);
            }
        }

        if (currentInput.Length != 1)
            throw new InvalidOperationException("Prediction should return only one matrix");
        return currentInput[0];
    }

    /// <summary>
    /// Saves feature maps to the specified directory.
    /// </summary>
    /// <param name="inputChannels"> Channels of input </param>
    /// <param name="directoryPath"></param>
    public void SaveFeatureMaps(Matrix[] inputChannels, string directoryPath)
    {
        Matrix[] currentInput = inputChannels;

        for (int i = 0; i < layers.Length; i++)
        {
            if (layers[i].LayerType == LayerType.Dropout)
            {
                float func = 1 - layersDropoutRates[layers[i]];
                currentInput = currentInput.Select(x => x * func).ToArray();
            }
            else
            {
                (currentInput, _) = layers[i].Forward(currentInput);
            }

            if (layers[i].LayerType == LayerType.Convolution || layers[i].LayerType == LayerType.Pooling)
            {
                for (int j = 0; j < currentInput.Length; j++)
                {
                    var featureMap = currentInput[j];

                    featureMap.SaveAsPng(directoryPath + $"featureMap_{i}_{j}.png");
                }
            }
        }
    }

    public float CalculateError((Matrix[] inputChannels, Matrix expectedOutput)[] testData)
    {
        float errorSum = 0;

        Parallel.ForEach(testData, item =>
        {
            var prediction = Predict(item.inputChannels);
            prediction = prediction + float.Epsilon;

            float error = ActivationFunctionsHandler.CalculateCrossEntropyCost(item.expectedOutput, prediction);
            errorSum += error;
        });

        return errorSum / testData.Length;
    }

    /// <summary>
    /// Calculates the correctness of the neural network on the given test data.
    /// </summary>
    /// <param name="testData"> Collection of data samples </param>
    /// <returns>Correctness in percent</returns>
    public float CalculateCorrectness((Matrix[] inputChannels, Matrix expectedOutput)[] testData)
    {
        int guessed = 0;

        Parallel.ForEach(testData, item =>
        {
            var prediction = Predict(item.inputChannels);
            var max = prediction.Max();

            int predictedNumber = prediction.IndexOfMax();
            int expectedNumber = item.expectedOutput.IndexOfMax();

            if (predictedNumber == expectedNumber)
            {
                Interlocked.Increment(ref guessed);
            }
        });

        return guessed * 100.0f / testData.Length;
    }

    #endregion INTERACTIONS

    #region SAVING / LOADING

    /// <summary>
    /// Saves the neural network to an XML file.
    /// </summary>
    /// <param name="path"> save file path </param>
    /// <param name="testCorrectness"> Test correcntess. Can be left as null. </param>
    /// <returns> True if success, false otherwise </returns>
    public bool SaveToXmlFile(string path, float? testCorrectness)
    {
        var writer = FilesCreatorHelper.CreateXmlFile(path);
        if (writer == null)
            return false;

        writer.WriteStartElement("Root");

        writer.WriteStartElement("Config");
        writer.WriteElementString("LearningRate", LearningRate.ToString());
        writer.WriteElementString("LayersAmount", layers.Length.ToString());
        writer.WriteElementString("LastTrainCorrectness", LastTrainCorrectness.ToString());
        if (testCorrectness != null)
            writer.WriteElementString("TestCorrectness", testCorrectness!.ToString());

        writer.WriteElementString("InputShape", $"{inputDepth} {inputRowsAmount} {inputColumnsAmount}");

        writer.WriteEndElement();

        writer.WriteStartElement("LayersHead");
        foreach (var layer in layers)
        {
            layer.SaveLayerDescription(writer);
        }
        writer.WriteEndElement();

        writer.WriteStartElement("LayersData");
        foreach (var layer in layers)
        {
            layer.SaveLayerData(writer);
        }
        writer.WriteEndElement();

        writer.WriteEndElement();
        writer.CloseXmlFile();

        return true;
    }

    /// <summary>
    /// Loads the neural network from an XML file.
    /// </summary>
    /// <param name="path"> Load file path </param>
    /// <returns> New object of <see cref="NeuralNetwork"/> if success, null otherwise </returns>
    public static NeuralNetwork? LoadFromXmlFile(string path)
    {
        if(File.Exists(path) == false)
            return null;

        XDocument xml = XDocument.Load(path);
        var root = xml.Root!;

        var config = root.Element("Config");
        if (config == null)
        {
            return null;
        }

        float learningRate = float.Parse(config.Element("LearningRate")!.Value);
        float lastTrainCorrectness = float.Parse(config.Element("LastTrainCorrectness")!.Value);
        int layersAmount = int.Parse(config.Element("LayersAmount")!.Value);

        string? inputShapeStr = config.Element("InputShape")?.Value;
        if (inputShapeStr == null)
            return null;

        string[] inputShape = inputShapeStr.Split(' ');
        if (inputShape.Length != 3 || !int.TryParse(inputShape[0], out int inputDepth) || !int.TryParse(inputShape[1], out int inputHeight) || !int.TryParse(inputShape[2], out int inputWidth))
            return null;

        var layersHead = root.Element("LayersHead")!.Elements();
        var layersData = root.Element("LayersData")!.Elements();

        Dictionary<ILayer, float> layersDropoutRates = new();

        ILayer[] layers = new ILayer[layersAmount];
        for (int i = 0; i < layersAmount; i++)
        {
            var layerHead = layersHead.ElementAt(i);
            var layerData = layersData.ElementAt(i);

            var layerTypeStr = layerHead.Attribute("LayerType")!.Value;
            LayerType layerType = Enum.Parse<LayerType>(layerTypeStr);

            ILayer? layer = null;
            switch (layerType)
            {
                case LayerType.Convolution:
                    layer = ConvolutionLayer.LoadLayerData(layerHead, layerData);
                    break;

                case LayerType.Pooling:
                    layer = PoolingLayer.LoadLayerData(layerHead, layerData);
                    break;

                case LayerType.FullyConnected:
                    layer = FullyConnectedLayer.LoadLayerData(layerHead, layerData);
                    break;

                case LayerType.Reshape:
                    layer = ReshapeFeatureToClassificationLayer.LoadLayerData(layerHead, layerData);
                    break;

                case LayerType.Dropout:
                    var dropLayer = DropoutLayer.LoadLayerData(layerHead, layerData);
                    if (dropLayer != null)
                    {
                        layer = dropLayer;
                        layersDropoutRates.Add(layer, dropLayer.dropoutRate);
                    }
                    break;

                default:
                    throw new ArgumentOutOfRangeException();
            }

            if (layer == null)
                return null;

            layers[i] = layer;
        }

        return new NeuralNetwork(inputDepth, inputHeight, inputWidth, layers, learningRate, lastTrainCorrectness, layersDropoutRates);
    }

    #endregion SAVING / LOADING

    #region FORWARD / BACKWARD

    /// <summary>
    /// Feeds the input channels through the neural network and returns the output.
    /// </summary>
    /// <param name="inputChannels"> Data sample (its channels) </param>
    /// <returns>
    /// Tuple with output matrix and array of layers outputs before activation functions
    /// </returns>
    /// <exception cref="InvalidOperationException"></exception>
    internal (Matrix output, (Matrix[] activated, Matrix[] beforeActivation)[] layersOutputs) Feedforward(Matrix[] inputChannels)
    {
        if (inputChannels.Length != inputDepth || inputChannels[0].RowsAmount != inputRowsAmount || inputChannels[0].ColumnsAmount != inputColumnsAmount)
            throw new InvalidOperationException($" Input channels have wrong dimensions!\n Was {inputChannels.Length}x{inputChannels[0].RowsAmount}x{inputChannels[0].ColumnsAmount} but expected {inputDepth}x{inputRowsAmount}x{inputColumnsAmount}");

        List<(Matrix[] activated, Matrix[] beforeActivation)> layersOutputs = new(this.layers.Length + 1);

        Matrix[] currentInput = inputChannels;
        layersOutputs.Add((currentInput, new Matrix[0]));

        for (int i = 0; i < layers.Length; i++)
        {
            (currentInput, var otherOutput) = layers[i].Forward(currentInput);
            layersOutputs.Add((currentInput, otherOutput));
        }

        if (currentInput.Length != 1)
            throw new InvalidOperationException("Prediction should return only one matrix");

        return (currentInput[0], layersOutputs.ToArray());
    }

    /// <summary>
    /// Backpropagates the error through the neural network.
    /// </summary>
    /// <param name="expectedResult">
    /// The expected result of the neural network.
    /// </param>
    /// <param name="prediction">
    /// Prediction of the neural network.
    /// </param>
    /// <param name="layersBeforeActivation">
    /// Collection of layers outputs before activation functions.
    /// </param>
    internal void Backpropagation(Matrix expectedResult, Matrix prediction, (Matrix[] activated, Matrix[] beforeActivation)[] layersOutputs)
    {
        var error = expectedResult.ElementWiseSubtract(prediction);

        Matrix[] currentError = [error];

        for (int i = layers.Length - 1; i >= 0; i--)
        {
            var thisLayerOutBeforeActivation = layersOutputs[i + 1].beforeActivation;
            var prevLayerOut = layersOutputs[i].activated;
            currentError = layers[i].Backward(currentError, prevLayerOut, thisLayerOutBeforeActivation, LearningRate);
        }
    }

    #endregion FORWARD / BACKWARD
}