using ToyNeuralNetwork.ImageProcessing;
using ToyNeuralNetwork.Math;
using NumSharp;
using System.Diagnostics;

namespace ToyNeuralNetwork.QuickDrawHandler;

public static class QuickDrawDataReader
{
    private static Random random = new Random();

    public static QuickDrawSet? LoadQuickDrawSamplesFromFiles(string[] filePaths, int amountToLoadFromEachFile = 2000, bool randomlyShift = true, bool colorReverse = true, float maxValue = 255.0f, CancellationToken ct = default, Action<int>? onFileLoaded = null)
    {
        var files = filePaths;

        Debug.WriteLine($"[LOADING SETS] Found {files.Length} files");

        List<IEnumerable<QuickDrawSample>> samples = new(filePaths.Length);

        int count = 0;
        foreach (var filePath in files)
        {
            if (ct.IsCancellationRequested)
                return null;

            var quickDrawSet = LoadDataFromNpyFile(filePath, amountToLoadFromEachFile, randomlyShift, colorReverse, maxValue, ct);

            if (ct.IsCancellationRequested)
                return null;

            samples.Add(quickDrawSet);

            count++;
            Debug.WriteLine($"[LOADING SETS] Loaded {count}/{files.Length} files");
            onFileLoaded?.Invoke(count);
        }

        return new QuickDrawSet(samples.ToArray());
    }

    public static QuickDrawSet? LoadQuickDrawSamplesFromDirectory(string directoryPath, int amountToLoadFromEachFile = 2000, bool randomlyShift = true, bool colorReverse = true, float maxValue = 255.0f, CancellationToken ct = default, Action<int>? onFileLoaded = null)
    {
        var files = Directory.GetFiles(directoryPath, "*.npy");

        return LoadQuickDrawSamplesFromFiles(files, amountToLoadFromEachFile, randomlyShift, colorReverse, maxValue, ct, onFileLoaded);
    }

    private static IEnumerable<QuickDrawSample> LoadDataFromNpyFile(string path, int amountToLoad, bool randomlyShift, bool colorReverse = true, float maxValue = 255.0f, CancellationToken ct = default)
    {
        NDArray npArray = np.load(path);
        float[,] array = (float[,])npArray.ToMuliDimArray<float>();

        QuickDrawSample[] quickDrawSamples = new QuickDrawSample[amountToLoad];
        string categoryName = Path.GetFileName(path.Replace(".npy", ""));

        int upperBound = amountToLoad > array.GetLength(0) ? array.GetLength(0) : amountToLoad;

        float factor = 1 / maxValue;

        Parallel.For(0, upperBound, (i, loopState) =>
        {
            if (ct.IsCancellationRequested)
            {
                loopState.Stop();
                return;
            }

            int sampleIndex = random.Next(array.GetLength(0));
            float[] data = new float[array.GetLength(1)];
            for (int j = 0; j < array.GetLength(1); j++)
            {
                data[j] = array[sampleIndex, j];
            }

            Matrix tmp = MatrixExtender.UnflattenMatrix(new Matrix(data), 28, 28)[0];
            tmp = colorReverse ? tmp.ApplyFunction(x => 1 - (x * factor)) : tmp * factor;

            if (randomlyShift)
            {
                float background = colorReverse ? 1.0f : 0.0f;
                tmp = ImageEditor.RandomShiftMatrixImage(tmp, 0.95f, 1.05f, -30f, 30f, background);
            }

            if (tmp.RowsAmount != 28 || tmp.ColumnsAmount != 28)
                throw new Exception("Matrix size is not 28x28");

            quickDrawSamples[i] = new QuickDrawSample(categoryName, [tmp]);
        });

        return quickDrawSamples;
    }
}