namespace ToyNeuralNetwork.Math;

public static class Statistics
{
    public static (float slope, float intercept) LinearRegression((float x, float y)[] points)
    {
        float sumX = 0;
        float sumY = 0;
        float sumXY = 0;
        float sumX2 = 0;
        float samples = points.Length;
        int n = points.Length;

        sumX = points.Sum(p => p.x);
        sumY = points.Sum(p => p.y);
        sumXY = points.Sum(p => p.x * p.y);
        sumX2 = points.Sum(p => p.x * p.x);

        float slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        float intercept = (sumY - slope * sumX) / n;

        return (slope, intercept);
    }

    public static (float slope, float intercept) LinearRegression(float[] x, float[] y)
    {
        if(x.Length != y.Length)
            throw new ArgumentException("x and y must have the same length");

       return LinearRegression(x.Zip(y).ToArray());
    }
}