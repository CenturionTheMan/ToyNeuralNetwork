using NeuralNetworkLibrary.Utils;

namespace NeuralNetworkLibrary.Math;

public static class ActivationFunctionsHandler
{
    public static Matrix ApplyActivationFunction(this Matrix input, ActivationFunction activationFunction)
    {
        switch (activationFunction)
        {
            case ActivationFunction.ReLU:
                return ActivationFunctionsHandler.ReLU(input);

            case ActivationFunction.Sigmoid:
                return ActivationFunctionsHandler.Sigmoid(input);

            case ActivationFunction.Softmax:
                return ActivationFunctionsHandler.Softmax(input);

            default:
                throw new ArgumentException("Invalid activation function");
        }
    }

    public static Matrix DerivativeActivationFunction(this Matrix input, ActivationFunction activationFunction)
    {
        switch (activationFunction)
        {
            case ActivationFunction.ReLU:
                return ActivationFunctionsHandler.DerivativeReLU(input);

            case ActivationFunction.Sigmoid:
                return ActivationFunctionsHandler.DerivativeSigmoid(input);

            case ActivationFunction.Softmax:
                return ActivationFunctionsHandler.DerivativeSoftmax(input);

            default:
                throw new ArgumentException("Invalid activation function");
        }
    }

    #region Activation Functions and Error

    /// <summary>
    /// Calculates the mean squared error between the expected and predicted results.
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="predictions"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static float CalculateMeanSquaredError(Matrix expected, Matrix predictions)
    {
        if (predictions.RowsAmount != expected.RowsAmount || predictions.ColumnsAmount != expected.ColumnsAmount)
        {
            throw new ArgumentException("Predictions and expected results matrices must have the same dimensions");
        }

        float sum = 0;

        for (int i = 0; i < predictions.RowsAmount; i++)
        {
            for (int j = 0; j < predictions.ColumnsAmount; j++)
            {
                sum += (float)System.Math.Pow(expected[i, j] - predictions[i, j], 2);
            }
        }

        return sum / (predictions.RowsAmount * predictions.ColumnsAmount);
    }

    /// <summary>
    /// Calculates the cross entropy cost between the expected and predicted results.
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="predictions"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static float CalculateCrossEntropyCost(Matrix expected, Matrix predictions)
    {
        if (predictions.RowsAmount != expected.RowsAmount || predictions.ColumnsAmount != expected.ColumnsAmount)
        {
            throw new ArgumentException("Predictions and expected results matrices must have the same dimensions");
        }

        float sum = 0;

        for (int i = 0; i < predictions.RowsAmount; i++)
        {
            for (int j = 0; j < predictions.ColumnsAmount; j++)
            {
                sum += expected[i, j] * (float)System.Math.Log(predictions[i, j]);
            }
        }

        return -sum;
    }

    /// <summary>
    /// Applies the ReLU activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    public static Matrix ReLU(Matrix mat)
    {
        return mat.ApplyFunction(x => { return x > 0 ? x : 0; });
    }

    /// <summary>
    /// Applies the derivative of the ReLU activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    public static Matrix DerivativeReLU(Matrix mat)
    {
        return mat.ApplyFunction(x => { return x >= 0 ? 1.0f : 0.0f; });
    }

    /// <summary>
    /// Applies the Sigmoid activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    public static Matrix Sigmoid(Matrix mat)
    {
        return mat.ApplyFunction(x => 1 / (float)(1 + System.Math.Exp(-x)));
    }

    /// <summary>
    /// Applies the derivative of the Sigmoid activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    public static Matrix DerivativeSigmoid(Matrix mat)
    {
        return mat.ApplyFunction(x =>
        {
            var sig = 1 / (1 + (float)System.Math.Exp(-x));
            return sig * (1 - sig);
        });
    }

    /// <summary>
    /// Applies the Softmax activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    public static Matrix Softmax(Matrix mat)
    {
        var expMat = mat.ApplyFunction(x => (float)System.Math.Exp(x));
        float sumOfMatrix = expMat.Sum() + float.Epsilon;
        var tmp = expMat.ApplyFunction(x => x / sumOfMatrix);
        return tmp;
    }

    /// <summary>
    /// Applies the derivative of the Softmax activation function to the given matrix.
    /// </summary>
    /// <param name="mat"></param>
    /// <returns></returns>
    public static Matrix DerivativeSoftmax(Matrix mat)
    {
        return Softmax(mat).ApplyFunction(x => x * (1 - x));
    }

    #endregion Activation Functions and Error
}