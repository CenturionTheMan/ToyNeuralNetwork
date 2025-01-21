using ToyNeuralNetwork.Math;

namespace NeuralNetworkUnitTest.Math;

public class MatrixTests
{
    #region BASE MATRIX

    [Fact]
    public void Constructor_SingleColumnValues_CreatesCorrectMatrix()
    {
        var matrix = new Matrix(new float[] { 1, 2, 3 });

        Assert.Equal(1, matrix[0, 0]);
        Assert.Equal(2, matrix[1, 0]);
        Assert.Equal(3, matrix[2, 0]);
    }

    [Fact]
    public void Constructor_2DArray_CreatesCorrectMatrix()
    {
        var matrix = new Matrix(new float[,] { { 1, 2 }, { 3, 4 } });

        Assert.Equal(1, matrix[0, 0]);
        Assert.Equal(2, matrix[0, 1]);
        Assert.Equal(3, matrix[1, 0]);
        Assert.Equal(4, matrix[1, 1]);
    }

    [Fact]
    public void Indexer_OutOfRange_ThrowsException()
    {
        var matrix = new Matrix(2, 2);
        Assert.Throws<IndexOutOfRangeException>(() => matrix[2, 2]);
    }

    [Fact]
    public void Indexer_InRange_ReturnsValid()
    {
        var matrix = new Matrix(2, 2);
        Assert.IsAssignableFrom<float>(matrix[1, 1]);
        Assert.IsAssignableFrom<float>(matrix[0, 0]);
        Assert.IsAssignableFrom<float>(matrix[1, 0]);
        Assert.IsAssignableFrom<float>(matrix[0, 1]);
    }

    [Fact]
    public void Enumerator_ValidMatrix_ReturnsCorrectValues()
    {
        var matrix = new Matrix(new float[,] {
            { 1.0f, 2.0f },
            { 3.0f, 4.0f } });
        var enumerator = matrix.GetEnumerator();

        Assert.True(enumerator.MoveNext());
        Assert.Equal(1.0, enumerator.Current);

        Assert.True(enumerator.MoveNext());
        Assert.Equal(2.0, enumerator.Current);

        Assert.True(enumerator.MoveNext());
        Assert.Equal(3.0, enumerator.Current);

        Assert.True(enumerator.MoveNext());
        Assert.Equal(4.0, enumerator.Current);

        Assert.False(enumerator.MoveNext());
    }

    [Fact]
    public void Equals_ValidMatrices_ReturnsCorrect()
    {
        Matrix matrix1 = new Matrix(new float[,]
        {
            { 1, 2 },
            { 3, 4 } });
        Matrix matrix2 = new Matrix(new float[,] {
            { 1, 2 },
            { 3, 4 } });

        Assert.True(matrix1.Equals(matrix2));
    }

    [Fact]
    public void Equals_InvalidMatrices_ReturnsCorrect()
    {
        Matrix matrix1 = new Matrix(new float[,]
        {
            { 1, 2 },
            { 3, 4 } });
        Matrix matrix2 = new Matrix(new float[,]
        {
            { 1, 2 },
            { 3, 5 } });

        Assert.False(matrix1.Equals(matrix2));
    }

    [Fact]
    public void Copy_ValidMatrix_ReturnsCorrectMatrix()
    {
        var matrix = new Matrix(new float[,]
        {
                { 1, 2 },
                { 3, 4 } });

        var copy = matrix.Copy();

        Assert.Equal(1, copy[0, 0]);
        Assert.Equal(2, copy[0, 1]);
        Assert.Equal(3, copy[1, 0]);
        Assert.Equal(4, copy[1, 1]);
    }

    [Fact]
    public void DotProductMatrices_ValidMatrices_ReturnsCorrectMatrix()
    {
        var matrix1 = new Matrix(new float[,] {
                { 1, 2 },
                { 3, 4 } });
        var matrix2 = new Matrix(new float[,] {
                { 2, 0 },
                { 1, 2 } });

        var result = Matrix.DotProductMatrices(matrix1, matrix2);

        Assert.Equal(4, result[0, 0]);
        Assert.Equal(4, result[0, 1]);
        Assert.Equal(10, result[1, 0]);
        Assert.Equal(8, result[1, 1]);
    }

    [Fact]
    public void DotProductMatrices_InvalidMatrices_ThrowsException()
    {
        var matrix1 = new Matrix(2, 2);
        var matrix2 = new Matrix(3, 3);

        Assert.Throws<ArgumentException>(() => Matrix.DotProductMatrices(matrix1, matrix2));
    }

    [Fact]
    public void ElementWiseMultiplyMatrices_ValidMatrices_ReturnsCorrectMatrix()
    {
        var matrix1 = new Matrix(new float[,] {
                { 1, 2 },
                { 3, 4 } });
        var matrix2 = new Matrix(new float[,] {
                { 2, 3 },
                { 4, 5 } });

        var result = Matrix.ElementWiseMultiplyMatrices(matrix1, matrix2);

        Assert.Equal(2, result[0, 0]);
        Assert.Equal(6, result[0, 1]);
        Assert.Equal(12, result[1, 0]);
        Assert.Equal(20, result[1, 1]);
    }

    [Fact]
    public void ElementWiseMultiplyMatrices_InvalidMatrices_ThrowsException()
    {
        var matrix1 = new Matrix(2, 2);
        var matrix2 = new Matrix(3, 3);

        Assert.Throws<ArgumentException>(() => Matrix.ElementWiseMultiplyMatrices(matrix1, matrix2));
    }

    [Fact]
    public void ElementWiseAddMatrices_ValidMatrices_ReturnsCorrectMatrix()
    {
        var matrix1 = new Matrix(new float[,]
        {
                { 1, 2 },
                { 3, 4 } });
        var matrix2 = new Matrix(new float[,]
        {
                { 2, 3 },
                { 4, 5 } });

        var result = Matrix.ElementWiseAddMatrices(matrix1, matrix2);

        Assert.Equal(3, result[0, 0]);
        Assert.Equal(5, result[0, 1]);
        Assert.Equal(7, result[1, 0]);
        Assert.Equal(9, result[1, 1]);
    }

    [Fact]
    public void ElementWiseAddMatrices_InvalidMatrices_ThrowsException()
    {
        var matrix1 = new Matrix(2, 2);
        var matrix2 = new Matrix(3, 3);

        Assert.Throws<ArgumentException>(() => Matrix.ElementWiseAddMatrices(matrix1, matrix2));
    }

    [Fact]
    public void ElementWiseSubtractMatrices_ValidMatrices_ReturnsCorrectMatrix()
    {
        var matrix1 = new Matrix(new float[,]
        {
                { 1, 2 },
                { 3, 4 } });
        var matrix2 = new Matrix(new float[,]
        {
                { 2, 3 },
                { 4, 5 } });

        var result = Matrix.ElementWiseSubtractMatrices(matrix1, matrix2);

        Assert.Equal(-1, result[0, 0]);
        Assert.Equal(-1, result[0, 1]);
        Assert.Equal(-1, result[1, 0]);
        Assert.Equal(-1, result[1, 1]);
    }

    [Fact]
    public void ElementWiseSubtractMatrices_InvalidMatrices_ThrowsException()
    {
        var matrix1 = new Matrix(2, 2);
        var matrix2 = new Matrix(3, 3);

        Assert.Throws<ArgumentException>(() => Matrix.ElementWiseSubtractMatrices(matrix1, matrix2));
    }

    [Fact]
    public void MultiplyOperator_ValidMatrices_ReturnsCorrectMatrix()
    {
        var matrix1 = new Matrix(new float[,]
        {
                { 1, 2 },
                { 3, 4 } });
        float numer = 2.5f;

        var result = matrix1 * numer;

        Assert.Equal(2.5, result[0, 0]);
        Assert.Equal(5, result[0, 1]);
        Assert.Equal(7.5, result[1, 0]);
        Assert.Equal(10, result[1, 1]);
    }

    [Fact]
    public void PlusOperator_ValidMatrices_ReturnsCorrectMatrix()
    {
        var matrix1 = new Matrix(new float[,]
        {
                { 1, 2 },
                { 3, 4 } });
        float numer = 15;

        var result = matrix1 + numer;

        Assert.Equal(16, result[0, 0]);
        Assert.Equal(17, result[0, 1]);
        Assert.Equal(18, result[1, 0]);
        Assert.Equal(19, result[1, 1]);
    }

    #endregion BASE MATRIX

    #region MATRIX EXTENSIONS CLASS

    [Fact]
    public void ConvolutionFull_ValidMatrix_ReturnsCorrectMatrix()
    {
        var input = new Matrix(new float[,]
        {
            {1, 2, 4, 3},
            {2, 1, 3, 5},
            {3, 2, 1, 6},
            {2, 3, 4, 9}
        });

        var kernel = new Matrix(new float[,]
        {
            {1,2,3 },
            {-4,3,-6 },
            {1,-2,3 },
        });
        int stride = 1;

        var expectedResult = new Matrix(new float[,]
        {
            { 1, 4, 11, 17, 18, 9},
            { -2, 0, -5, 2, 4, -3},
            { -4, 10, -4, -2, 18, -3},
            { -8, 5, 7, -5, 41, 6},
            { -5,-10,-13,-32,-6,-36},
            { 2,-1,4,10,-6,27},
        });

        var result = input.ConvolutionFull(kernel, stride);

        Assert.True(expectedResult.Equals(result));
    }

    [Fact]
    public void ConvolutionValid_ValidMatrix_ReturnsCorrectMatrix()
    {
        var input = new Matrix(new float[,]
        {
            {1, 2, 4, 3},
            {2, 1, 3, 5},
            {3, 2, 1, 6},
            {2, 3, 4, 9}
        });

        var kernel = new Matrix(new float[,]
        {
            {1,2,3 },
            {-4,3,-6 },
            {1,-2,3 },
        });
        int stride = 1;

        var expectedResult = new Matrix(new float[,]
        {
            { -4, -2 },
            { 7, -5 }
        });

        var result = input.ConvolutionValid(kernel, stride);

        Assert.True(expectedResult.Equals(result));
    }

    [Fact]
    public void CrossCorrelationFull_ValidMatrix_ReturnsCorrectMatrix()
    {
        var input = new Matrix(new float[,]
        {
            {1, 2, 4, 3},
            {2, 1, 3, 5},
            {3, 2, 1, 6},
            {2, 3, 4, 9}
        });

        var kernel = new Matrix(new float[,]
        {
            {1,2,3 },
            {-4,3,-6 },
            {1,-2,3 },
        });
        int stride = 1;

        var expectedResult = new Matrix(new float[,]
        {
            { 3, 4, 9, 3, -2, 3},
            { 0, -10, -13, -4, -14, -7},
            { 0, 8, -4, 12, 2, -11},
            { -6, 9, 9, 3, 13, -10},
            { -3, 0, -13, -32, 24, -30},
            { 6, 13, 20, 38, 22, 9},
        });

        var result = input.CrossCorrelationFull(kernel, stride);

        Assert.True(expectedResult.Equals(result));
    }

    [Fact]
    public void CrossCorrelationValid_ValidMatrix_ReturnsCorrectMatrix()
    {
        var input = new Matrix(new float[,]
        {
            {1, 2, 4, 3},
            {2, 1, 3, 5},
            {3, 2, 1, 6},
            {2, 3, 4, 9}
        });

        var kernel = new Matrix(new float[,]
        {
            {1,2,3 },
            {-4,3,-6 },
            {1,-2,3 },
        });
        int stride = 1;

        var expectedResult = new Matrix(new float[,]
        {
            {-4, 12 },
            {9,3 }
        });

        var result = input.CrossCorrelationValid(kernel, stride);

        Assert.True(expectedResult.Equals(result));
    }

    [Fact]
    public void Rotate180_ValidMatrix_ReturnsCorrectMatrix()
    {
        var matrix = new Matrix(new float[,]
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        var expectedResult = new Matrix(new float[,]
        {
            {9, 8, 7},
            {6, 5, 4},
            {3, 2, 1}
        });

        var result = matrix.Rotate180();

        Assert.True(expectedResult.Equals(result));
    }

    [Fact]
    public void IndexOfMax_ValidMatrix_ReturnsCorrectIndex()
    {
        var matrix = new Matrix([1, 2, 7, 2, 423, 4, 32, 1]);

        Assert.Equal(4, matrix.IndexOfMax());
    }

    [Fact]
    public void IndexOfMax_InvalidMatrix_ThrowsArgumentException()
    {
        var matrix = new Matrix(new float[,]
        {
            { 1, 2, 3 },
            { 4, 5, 30 },
            { 7, 8, 9 }
        });

        Assert.Throws<ArgumentException>(() => matrix.IndexOfMax());
    }

    [Fact]
    public void Transpose_ValidMatrix_ReturnsCorrectMatrix()
    {
        var matrix = new Matrix(new float[,]
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        var expectedResult = new Matrix(new float[,]
        {
            {1, 4, 7},
            {2, 5, 8},
            {3, 6, 9}
        });

        var result = matrix.Transpose();

        Assert.True(expectedResult.Equals(result));
    }

    #endregion MATRIX EXTENSIONS CLASS
}