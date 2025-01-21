
namespace ToyNeuralNetwork.Math;

public static class MatrixExtender
{
    #region SHAPE TRANSFORMATIONS

    public static (Matrix result, Matrix maxIndexMap) MaxPooling(this Matrix matrix, int poolSize, int stride)
    {
        int newRows = (matrix.RowsAmount - poolSize) / stride + 1;
        int newColumns = (matrix.ColumnsAmount - poolSize) / stride + 1;
        Matrix result = new Matrix(newRows, newColumns);
        Matrix maxIndexMap = new Matrix(newRows, newColumns);

        for (int i = 0; i < newRows; i++)
        {
            for (int j = 0; j < newColumns; j++)
            {
                float max = float.MinValue;
                int maxIndex = -1;
                for (int x = 0; x < poolSize; x++)
                {
                    for (int y = 0; y < poolSize; y++)
                    {
                        int rowIndex = i * stride + x;
                        int colIndex = j * stride + y;
                        if (rowIndex < matrix.RowsAmount && colIndex < matrix.ColumnsAmount && matrix[rowIndex, colIndex] > max)
                        {
                            max = matrix[rowIndex, colIndex];
                            maxIndex = MatrixHelpers.GetIndex(rowIndex, colIndex, matrix.ColumnsAmount);
                        }
                    }
                }
                result[i, j] = max;
                maxIndexMap[i, j] = maxIndex;
            }
        }

        return (result, maxIndexMap);
    }

    public static (int outputRows, int outputColumns) GetSizeAfterPooling((int rows, int columns) inputSize, int poolSize, int stride)
    {
        return GetSizeAfterConvolution(inputSize, (poolSize, poolSize), stride);
    }

    public static (int outputRows, int outputColumns) GetSizeAfterConvolution((int rows, int columns) inputSize, (int rows, int columns) kernel, int stride)
    {
        var outputRows = (inputSize.rows - kernel.rows) / stride + 1;
        var outputColumns = (inputSize.columns - kernel.columns) / stride + 1;
        return (outputRows, outputColumns);
    }

    public static Matrix FlattenMatrix(params Matrix[] matrices)
    {
        List<float> flattenedList = new();
        foreach (var matrix in matrices)
        {
            for (int i = 0; i < matrix.RowsAmount; i++)
            {
                for (int j = 0; j < matrix.ColumnsAmount; j++)
                {
                    flattenedList.Add(matrix[i, j]);
                }
            }
        }

        return new Matrix(flattenedList.ToArray());
    }

    public static Matrix[] UnflattenMatrix(Matrix flattenedMatrix, int rowsAmount, int columnsAmount)
    {
        if (flattenedMatrix.RowsAmount % rowsAmount * columnsAmount != 0)
            throw new ArgumentException("Invalid matrix size");

        List<Matrix> matrices = new List<Matrix>();

        int index = 0;
        for (int i = 0; i < flattenedMatrix.RowsAmount; i += rowsAmount * columnsAmount)
        {
            Matrix matrix = new Matrix(rowsAmount, columnsAmount);
            for (int j = 0; j < rowsAmount; j++)
            {
                for (int k = 0; k < columnsAmount; k++)
                {
                    matrix[j, k] = flattenedMatrix[index++, 0];
                }
            }
            matrices.Add(matrix);
        }

        return matrices.ToArray();
    }

    public static Matrix[] UnflattenMatrix(Matrix flattenedMatrix, int matrixSize)
    {
        return UnflattenMatrix(flattenedMatrix, matrixSize, matrixSize);
    }

    #endregion SHAPE TRANSFORMATIONS


    #region CONVOLUTION / CROSS-CORRELATION

    public static Matrix ConvolutionFull(this Matrix input, Matrix kernel, int stride)
    {
        int outputRows = input.RowsAmount + kernel.RowsAmount - 1;
        int outputColumns = input.ColumnsAmount + kernel.ColumnsAmount - 1;
        Matrix output = new Matrix(outputRows, outputColumns);

        for (int i = 0; i < outputRows; i++)
        {
            for (int j = 0; j < outputColumns; j++)
            {
                float sum = 0;

                for (int m = 0; m < kernel.RowsAmount; m++)
                {
                    for (int n = 0; n < kernel.ColumnsAmount; n++)
                    {
                        int x = i * stride - m;
                        int y = j * stride - n;

                        if (x >= 0 && x < input.RowsAmount && y >= 0 && y < input.ColumnsAmount)
                        {
                            sum += input[x, y] * kernel[m, n];
                        }
                    }
                }

                output[i, j] = sum;
            }
        }
        return output;
    }

    public static Matrix ConvolutionValid(this Matrix input, Matrix kernel, int stride)
    {
        var rot = kernel.Rotate180();
        return input.CrossCorrelationValid(rot, stride);
    }

    public static Matrix CrossCorrelationFull(this Matrix input, Matrix kernel, int stride)
    {
        var rot = kernel.Rotate180();
        return input.ConvolutionFull(rot, stride);
    }

    public static Matrix CrossCorrelationValid(this Matrix input, Matrix kernel, int stride)
    {
        int outputRows = (input.RowsAmount - kernel.RowsAmount) / stride + 1;
        int outputColumns = (input.ColumnsAmount - kernel.ColumnsAmount) / stride + 1;
        Matrix output = new Matrix(outputRows, outputColumns);

        for (int i = 0; i < outputRows; i++)
        {
            for (int j = 0; j < outputColumns; j++)
            {
                float sum = 0;

                for (int m = 0; m < kernel.RowsAmount; m++)
                {
                    for (int n = 0; n < kernel.ColumnsAmount; n++)
                    {
                        int x = i * stride + m;
                        int y = j * stride + n;

                        if (x >= 0 && x < input.RowsAmount && y >= 0 && y < input.ColumnsAmount)
                        {
                            sum += input[x, y] * kernel[m, n];
                        }
                    }
                }

                output[i, j] = sum;
            }
        }
        return output;
    }

    public static Matrix AddPadding(this Matrix input, int padding)
    {
        Matrix output = new Matrix(input.RowsAmount + 2 * padding, input.ColumnsAmount + 2 * padding);

        for (int i = 0; i < input.RowsAmount; i++)
        {
            for (int j = 0; j < input.ColumnsAmount; j++)
            {
                output[i + padding, j + padding] = input[i, j];
            }
        }

        return output;
    }

    /// <summary>
    /// Rotates the matrix by 180 degrees
    /// </summary>
    /// <returns>New matrix after rotation</returns>
    public static Matrix Rotate180(this Matrix matrix)
    {
        Matrix result = new Matrix(matrix.RowsAmount, matrix.ColumnsAmount);
        for (int i = 0; i < matrix.RowsAmount; i++)
        {
            for (int j = 0; j < matrix.ColumnsAmount; j++)
            {
                result[i, j] = matrix[matrix.RowsAmount - i - 1, matrix.ColumnsAmount - j - 1];
            }
        }
        return result;
    }

    #endregion CONVOLUTION / CROSS-CORRELATION


    #region MATH OPERATIONS

     public static Matrix ClampValues(this Matrix a, float min, float max)
    {
        return a.ApplyFunction(x => System.Math.Clamp(x, min, max));
    }

    public static int IndexOfMax(this Matrix matrix)
    {
        if (matrix.ColumnsAmount != 1)
            throw new ArgumentException("Matrix must have only one column");

        float max = float.MinValue;
        int index = 0;
        for (int i = 0; i < matrix.RowsAmount; i++)
        {
            if (matrix[i, 0] > max)
            {
                max = matrix[i, 0];
                index = i;
            }
        }
        return index;
    }

    /// <summary>
    /// Sums all elements of the matrix
    /// </summary>
    /// <param name="a">Matrix to sum</param>
    /// <returns>Sum of all elements</returns>
    public static float Sum(this Matrix a)
    {
        float sum = 0;
        foreach (var item in a)
        {
            sum += item;
        }
        return sum;
    }

    /// <summary>
    /// Finds the maximum value in the matrix
    /// </summary>
    /// <param name="a">Matrix to search</param>
    /// <returns>Maximum value in the matrix</returns>
    public static float Max(this Matrix a)
    {
        float max = float.MinValue;
        foreach (var item in a)
        {
            if (item > max)
            {
                max = item;
            }
        }
        return max;
    }

    /// <summary>
    /// Transposes the matrix
    /// </summary>
    /// <param name="a">Matrix to transpose</param>
    /// <returns>Transposed matrix</returns>
    public static Matrix Transpose(this Matrix a)
    {
        Matrix result = new Matrix(a.ColumnsAmount, a.RowsAmount);

        for (int i = 0; i < a.ColumnsAmount; i++)
        {
            for (int j = 0; j < a.RowsAmount; j++)
            {
                result[i, j] = a[j, i];
            }
        }

        return result;
    }

    /// <summary>
    /// Applies the given function to each element of the matrix
    /// </summary>
    /// <param name="a">Matrix to apply function to</param>
    /// <param name="function">Function to apply</param>
    /// <returns>Matrix with applied function</returns>
    public static Matrix ApplyFunction(this Matrix a, Func<float, float> function)
    {
        Matrix result = new Matrix(a.RowsAmount, a.ColumnsAmount);

        for (int i = 0; i < a.RowsAmount; i++)
        {
            for (int j = 0; j < a.ColumnsAmount; j++)
            {
                result[i, j] = function(a[i, j]);
            }
        }
        return result;
    }

    public static Matrix DotProduct(this Matrix a, Matrix b)
    {
        return Matrix.DotProductMatrices(a, b);
    }

    public static Matrix ElementWiseMultiply(this Matrix a, Matrix b)
    {
        return Matrix.ElementWiseMultiplyMatrices(a, b);
    }

    public static Matrix ElementWiseAdd(this Matrix a, Matrix b)
    {
        return Matrix.ElementWiseAddMatrices(a, b);
    }

    public static Matrix ElementWiseSubtract(this Matrix a, Matrix b)
    {
        return Matrix.ElementWiseSubtractMatrices(a, b);
    }

    #endregion MATH OPERATIONS
    
}