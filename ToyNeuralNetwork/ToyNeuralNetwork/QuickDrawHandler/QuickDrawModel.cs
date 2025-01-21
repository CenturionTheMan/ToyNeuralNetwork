using NeuralNetworkLibrary.Math;

namespace NeuralNetworkLibrary.QuickDrawHandler;

public struct QuickDrawSample
{
    public readonly string category;
    public readonly Matrix[] data;

    public QuickDrawSample(string category, Matrix[] data)
    {
        this.category = category;
        this.data = data;
    }
}

public class QuickDrawSet
{
    private static Random random = new();

    public static readonly Dictionary<string, int> CategoryToIndex = new Dictionary<string, int>{
        { "axe", 0},
        { "cactus", 1},
        { "cat", 2},
        { "diamond", 3},
        { "moustache", 4},
        { "pants", 5},
        { "snowman", 6},
        { "stairs", 7},
        { "sword", 8},
    };

    public static readonly Dictionary<int, string> IndexToCategory = CategoryToIndex.ToDictionary(x => x.Value, x => x.Key);

    public readonly IEnumerable<QuickDrawSample>[] SamplesByCategory;
    public IEnumerable<QuickDrawSample> AllSamples => SamplesByCategory.SelectMany(x => x).OrderBy(x => random.Next());

    internal QuickDrawSet(IEnumerable<QuickDrawSample>[] samples)
    {
        this.SamplesByCategory = samples;
    }

    private Matrix OutputForNN(string category)
    {
        float[] output = new float[9];
        output[CategoryToIndex[category]] = 1;

        return new Matrix(output);
    }

    public ((Matrix[] inputs, Matrix outputs)[] trainData, (Matrix[] inputs, Matrix outputs)[] testData) SplitIntoTrainTest(int testSizePercent = 20)
    {
        int allDataLen = SamplesByCategory[0].Count() * SamplesByCategory.Length;
        int testCount = (int)(allDataLen * (testSizePercent / 100.0));

        List<QuickDrawSample> trainData = new(allDataLen - testCount);
        List<QuickDrawSample> testData = new(testCount);

        foreach (var catSamples in SamplesByCategory)
        {
            int catTestCount = (int)(catSamples.Count() * (testSizePercent / 100.0));
            var subTest = catSamples.Take(catTestCount);
            var subTrain = catSamples.Skip(catTestCount);

            trainData.AddRange(subTrain);
            testData.AddRange(subTest);
        }

        trainData = trainData.OrderBy(x => random.Next()).ToList();
        testData = testData.OrderBy(x => random.Next()).ToList();

        var train = trainData.Select(x => (x.data, OutputForNN(x.category))).ToArray();
        var test = testData.Select(x => (x.data, OutputForNN(x.category))).ToArray();

        return (train, test);
    }

    public ((Matrix[] inputs, Matrix outputs)[] trainData, (Matrix[] inputs, Matrix outputs)[] testData) SplitIntoTrainTestFlattenInput(int testSizePercent = 20)
    {
        (var trainData, var testData) = SplitIntoTrainTest(testSizePercent);
        var trainFlat = trainData.Select(i => (new Matrix[] { MatrixExtender.FlattenMatrix(i.inputs) }, i.outputs));
        var testFlat = testData.Select(i => (new Matrix[] { MatrixExtender.FlattenMatrix(i.inputs) }, i.outputs));

        return (trainFlat.ToArray(), testFlat.ToArray());
    }
}