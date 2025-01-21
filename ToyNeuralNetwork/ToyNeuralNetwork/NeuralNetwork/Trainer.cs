using ToyNeuralNetwork.Math;
using ToyNeuralNetwork.Utils;
using System.Diagnostics;

namespace ToyNeuralNetwork.NeuralNetwork;

public class Trainer
{
    public NeuralNetwork NeuralNetwork => neuralNetwork;

    private NeuralNetwork neuralNetwork;
    private (Matrix[] inputChannels, Matrix output)[] data;
    private float initialLearningRate;
    private float minLearningRate;
    private int epochAmount;
    private int batchSize;

    private bool isPatience = false;
    private float userInitialIgnore = 0.0f;
    private float userPatience = 0;
    private float initialIgnore = 10.0f; //percent
    private int patienceAmount = 50;
    private bool hasCrossedIgnoreThreshold = false;
    private Func<float, int, float> learningRateModifier = (lr, epoch) => lr * 0.9f;

    private bool saveToLog = false;
    private bool saveNN = false;
    private (Matrix[] inputChannels, Matrix output)[]? testData;
    private string trainingLogDir = "";

    private record TrainingIterationData(int epoch, int dataIndex, float error, float learningRate, float elapsedSeconds);
    private record TrainingBatchData(int epoch, float avgBatchError, float learningRate, float elapsedSeconds);

    /// <summary>
    /// Create a new instance of the <see cref="Trainer"/> class.
    /// </summary>
    /// <param name="neuralNetwork">
    /// Neural network to train.
    /// </param>
    /// <param name="data">
    /// Data to train on.
    /// </param>
    /// <param name="initialLearningRate">
    /// Initial learning rate.
    /// </param>
    /// <param name="minLearningRate">
    /// Minimum learning rate. Meeting this value will stop the training with first epoch end.
    /// </param>
    /// <param name="epochAmount">
    /// Maximum amount of epochs to train.
    /// </param>
    /// <param name="batchSize">
    /// Batch size.
    /// </param>
    public Trainer(NeuralNetwork neuralNetwork, (Matrix[] inputChannels, Matrix output)[] data, float initialLearningRate, float minLearningRate, int epochAmount, int batchSize)
    {
        this.neuralNetwork = neuralNetwork;
        this.data = data;
        this.initialLearningRate = initialLearningRate;
        this.minLearningRate = minLearningRate;
        this.epochAmount = epochAmount;
        this.batchSize = batchSize;
    }

    /// <summary>
    /// Set patience for the training.
    /// </summary>
    /// <param name="initialIgnore">
    /// Initial ignore threshold. Value must be in range [0, 0.95].
    /// Learning rate won t be modified until this threshold is crossed.
    /// 0.0 means that learning rate will be modified from the beginning.
    /// 0.95 means that learning rate will be modified when 95% of first epoch is finished.
    /// </param>
    /// <param name="patience">
    /// Patience is the amount of data that will be used to calculate the slope of the error.
    /// If slope is zero or positive, learning rate will be decreased.
    /// Value must be in range [0, 1].
    /// 0.0 means that learning rate may be modified after each batch.
    /// 1.0 means that learning rate may be modified after each epoch.
    /// </param>
    /// <param name="learningRateModifier">
    /// Modifier for the learning rate. If null, default modifier will be used.
    /// Default modifier is: lr => lr * 0.9f
    /// </param>
    /// <returns>
    /// Trainer instance.
    /// </returns>
    public Trainer SetPatience(float initialIgnore, float patience, Func<float, int, float>? learningRateModifier = null)
    {
        initialIgnore = System.Math.Clamp(initialIgnore, 0, 0.95f);
        patience = System.Math.Clamp(patience, 0, 1f);

        this.userInitialIgnore = initialIgnore;
        this.userPatience = patience;

        this.initialIgnore = initialIgnore * 100;
        this.patienceAmount = (int)(data.Length * patience / batchSize);
        this.patienceAmount = System.Math.Max(1, patienceAmount);

        if (learningRateModifier is not null)
            this.learningRateModifier = learningRateModifier;

        isPatience = true;

        return this;
    }

    /// <summary>
    /// Set saving of the training data.
    /// </summary>
    /// <param name="outputDirPath">
    /// Output directory path. If directory does not exist, it will be created. Logs will be saved in subdirectory with current date and time.
    /// </param>
    /// <param name="saveNN">
    /// Flag determining if neural network should be saved.
    /// </param>
    /// <param name="trainingLogDirectory">
    /// Directory where logs will be saved.
    /// </param>
    /// <returns>
    /// Trainer instance.
    /// </returns>
    public Trainer SetLogSaving(string outputDirPath, bool saveNN, (Matrix[] inputChannels, Matrix output)[]? testData, out string trainingLogDirectory)
    {
        if (!Directory.Exists(outputDirPath))
            Directory.CreateDirectory(outputDirPath);

        this.trainingLogDir = outputDirPath + DateTime.Now.ToString("yyyy.MM.dd__HH-mm-ss");

        this.testData = testData;

        if (!Directory.Exists(trainingLogDir))
            Directory.CreateDirectory(trainingLogDir);
        this.trainingLogDir += "/";

        this.saveNN = saveNN;
        this.saveToLog = true;

        trainingLogDirectory = this.trainingLogDir;
        return this;
    }

    /// <summary>
    /// Run the training.
    /// </summary>
    public (Task, CancellationTokenSource) RunTrainingOnTask()
    {
        var stopwatch = new Stopwatch();

        var cts = new CancellationTokenSource();

        Queue<TrainingIterationData> trainingIterationData = new(epochAmount * data.Length);
        Queue<TrainingBatchData> trainingBatchData = new(data.Length * epochAmount / batchSize);

        List<(int, float, float)> trainCorrectness = new(epochAmount + 1);
        Queue<(float error, float seconds)>? lastBatchErrors = new(patienceAmount);


        neuralNetwork.OnTrainingIteration += (epoch, dataIndex, error) =>
        {
            if (saveToLog)
            {
                var elapsedSeconds = stopwatch.Elapsed.TotalSeconds;
                trainingIterationData.Enqueue(new TrainingIterationData(epoch, dataIndex, error, neuralNetwork.LearningRate, (float)elapsedSeconds));
            }

            if (float.IsNaN(error))
            {
                cts?.Cancel();
            }
        };

        neuralNetwork.OnBatchTrainingIteration += (epoch, epochPercentFinish, batchAvgError) =>
        {
            if (isPatience)
            {
                lastBatchErrors!.Enqueue((batchAvgError, (float)stopwatch.Elapsed.TotalSeconds));
                HandlePatience(neuralNetwork, lastBatchErrors, epochPercentFinish, epoch);
            }

            if (saveToLog)
            {
                var time = stopwatch.Elapsed.TotalSeconds;
                trainingBatchData.Enqueue(new TrainingBatchData(epoch, batchAvgError, neuralNetwork.LearningRate, (float)time));
            }
        };

        var epochLogData = testData is null ? data.Take(1000).ToArray() : testData;
        
        if(saveToLog)
            trainCorrectness.Add((0, neuralNetwork.CalculateCorrectness(epochLogData), neuralNetwork.CalculateError(epochLogData)));
        neuralNetwork.OnEpochTrainingIteration += (epoch, _) =>
        {
            if(saveToLog)
            {
                var correctness = neuralNetwork.CalculateCorrectness(epochLogData);
                var error = neuralNetwork.CalculateError(epochLogData);
                trainCorrectness.Add((epoch, correctness, error));
            }

            if (neuralNetwork.LearningRate <= minLearningRate)
            {
                cts?.Cancel();
            }
        };

        Thread consoleThread = new Thread(() =>
        {
            while (cts!.IsCancellationRequested == false)
            {
                var pressedKey = Console.ReadLine();
                if (pressedKey?.ToLower() == "q")
                {
                    cts!.Cancel();
                }
            }
        });
        consoleThread.Start();

        neuralNetwork.OnTrainingFinished += () =>
        {
            if (saveToLog)
                SaveTrainingData(trainingLogDir, trainingIterationData.ToArray(), trainingBatchData.ToArray(), trainCorrectness.ToArray());

            stopwatch.Stop();
        };

        stopwatch.Start();
        var task = neuralNetwork.TrainOnNewTask(data, initialLearningRate, epochAmount, batchSize, cts.Token);
        return (task, cts);
    }

    /// <summary>
    /// Handle patience mechanism.
    /// </summary>
    /// <param name="nn">
    /// Neural network.
    /// </param>
    /// <param name="lastAvgBatchErrors">
    /// Last batch errors.
    /// </param>
    /// <param name="epochPercentFinish">
    /// Percent of epoch finish.
    /// </param>
    private void HandlePatience(NeuralNetwork nn, Queue<(float, float)> lastAvgBatchErrors, float epochPercentFinish, int epoch)
    {
        if (hasCrossedIgnoreThreshold == false || nn.LearningRate <= minLearningRate)
        {
            if (epochPercentFinish >= initialIgnore)
                hasCrossedIgnoreThreshold = true;

            return;
        }

        if (lastAvgBatchErrors.Count() < patienceAmount) return;

        (float slope, _) = Statistics.LinearRegression(lastAvgBatchErrors.ToArray());

        if (slope >= 0)
        {
            nn.LearningRate = System.Math.Max(minLearningRate, learningRateModifier(nn.LearningRate, epoch));
        }
        lastAvgBatchErrors!.Clear();
    }

    /// <summary>
    /// Save training data.
    /// </summary>
    /// <param name="dirPath">
    /// Directory path.
    /// </param>
    /// <param name="trainingIterationData">
    /// Data from training iterations.
    /// </param>
    /// <param name="trainEpochCorrectness">
    /// Epoch correctness.
    /// </param>
    private void SaveTrainingData(string dirPath, TrainingIterationData[] trainingIterationData, TrainingBatchData[] trainingBatchData, (int, float, float)[] trainEpochCorrectness)
    {
        trainingIterationData = trainingIterationData.Where(x => x is not null).ToArray();
        trainingBatchData = trainingBatchData.Where(x => x is not null).ToArray();

        List<object[]> data = [["Epoch", "DataIndex", "Error", "LearningRate", "ElapsedSeconds"]];
        foreach (var item in trainingIterationData)
        {
            data.Add([item.epoch, item.dataIndex, item.error, item.learningRate, item.elapsedSeconds]);
        }
        FilesCreatorHelper.CreateCsvFile(data, dirPath + "AllErrors.csv");
        data.Clear();

        data = [["Epoch", "AvgBatchError", "LearningRate", "ElapsedSeconds"]];
        foreach (var item in trainingBatchData)
        {
            data.Add([item.epoch, item.avgBatchError, item.learningRate, item.elapsedSeconds]);
        }
        FilesCreatorHelper.CreateCsvFile(data, dirPath + "BatchError.csv");
        data.Clear();

        data = [["Epoch", "Correctness", "TestError", "AvgTrainError", "ElapsedSeconds"]];
        int dataLength = trainingIterationData.Length / trainEpochCorrectness.Length;
        for (int i = 0; i < trainEpochCorrectness.Length; i++)
        {
            var tmp = trainingIterationData.Where(x => x.epoch == i).ToArray();

            string avgError = tmp.Count() > 0 ? tmp.Average(x => x.error).ToString() : "null";
            string elapsedSeconds = tmp.Count() > 0 ? tmp.Max(x => x.elapsedSeconds).ToString() : "0";

            data.Add([i, trainEpochCorrectness[i].Item2, trainEpochCorrectness[i].Item3, avgError, elapsedSeconds]);
        }
        FilesCreatorHelper.CreateCsvFile(data, dirPath + "EpochError.csv");
        data.Clear();

        if (saveNN)
        {
            float? cor = testData is null ? null : neuralNetwork.CalculateCorrectness(testData);
            neuralNetwork.SaveToXmlFile(dirPath + "NeuralNetwork.xml", cor);
        }

        var xml = FilesCreatorHelper.CreateXmlFile(dirPath + "TrainerConfig.xml");
        if (xml is not null)
        {
            xml.WriteStartElement("Root");
            xml.WriteStartElement("BaseConfig");
            xml.WriteElementString("InitialLearningRate", initialLearningRate.ToString());
            xml.WriteElementString("EpochAmount", epochAmount.ToString());
            xml.WriteElementString("BatchSize", batchSize.ToString());
            xml.WriteEndElement();

            if (isPatience)
            {
                xml.WriteStartElement("PatienceConfig");
                xml.WriteElementString("InitialIgnore", userInitialIgnore.ToString());
                xml.WriteElementString("Patience", userPatience.ToString());
                xml.WriteEndElement();
            }
            xml.WriteEndElement();
            FilesCreatorHelper.CloseXmlFile(xml);
        }

        data = [["ClassIndex", "ClassName", "TestCorrecntess", "TrainCorrectness"]];
        int classesAmount = this.data.First().output.RowsAmount;
        for (int i = 0; i < classesAmount; i++)
        {
            if (!QuickDrawHandler.QuickDrawSet.IndexToCategory.TryGetValue(i, out string? className))
                className = "Unknown";

            var testCorrectness = testData is null ? "null" : neuralNetwork.CalculateCorrectness(testData.Where(s => s.output[i, 0] == 1.0f).ToArray()).ToString("0.000") + "%";
            var trainCorrectness = neuralNetwork.CalculateCorrectness(this.data.Where(s => s.output[i, 0] == 1.0f).Take(1000).ToArray()).ToString("0.000") + "%";

            data.Add([i, className, testCorrectness, trainCorrectness]);
        }
        FilesCreatorHelper.CreateCsvFile(data, dirPath + "ClassCorrectness.csv");
        data.Clear();
    }
}