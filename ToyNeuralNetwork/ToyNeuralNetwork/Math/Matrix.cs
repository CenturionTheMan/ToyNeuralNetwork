using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary.Math;

public class Matrix
{
    private static Random random = new();
    private float[,] values;

    public readonly int RowsAmount;
    public readonly int ColumnsAmount;

    /// <summary>
    /// Creates a new Matrix filled with zeros of dimensions rows: singleColumnValues, columns: 1
    /// </summary>
    /// <param name="singleColumnValues">Rows amount</param>
    public Matrix(float[] singleColumnValues)
    {
        this.values = new float[singleColumnValues.Length, 1];

        for (int i = 0; i < singleColumnValues.Length; i++)
        {
            values[i, 0] = singleColumnValues[i];
        }

        RowsAmount = singleColumnValues.Length;
        ColumnsAmount = 1;
    }

    /// <summary>
    /// Creates a new Matrix based on the given values
    /// </summary>
    /// <param name="values">values to be assigned to the matrix</param>
    public Matrix(float[,] values)
    {
        this.values = values;
        RowsAmount = values.GetLength(0);
        ColumnsAmount = values.GetLength(1);
    }

    /// <summary>
    /// Creates a new Matrix filled with zeros of dimensions rows: rowsAmount, columns: columnsAmount
    /// </summary>
    /// <param name="rowsAmount">Rows amount</param>
    /// <param name="columnsAmount">Columns amount</param>
    public Matrix(int rowsAmount, int columnsAmount)
    {
        values = new float[rowsAmount, columnsAmount];
        RowsAmount = rowsAmount;
        ColumnsAmount = columnsAmount;
    }

    /// <summary>
    /// Creates a new Matrix filled with random values of dimensions rows: rowsAmount, columns: columnsAmount
    /// </summary>
    /// <param name="rowsAmount">Rows amount</param>
    /// <param name="columnsAmount">Columns amount</param>
    /// <param name="min">Minimum value</param>
    /// <param name="max">Maximum value</param>
    public Matrix(int rowsAmount, int columnsAmount, float min, float max)
    {
        values = new float[rowsAmount, columnsAmount];
        RowsAmount = rowsAmount;
        ColumnsAmount = columnsAmount;

        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                values[i, j] = random.NextSingle() * (max - min) + min;
            }
        }
    }

    public void InitializeXavier()
    {
        float limit = (float)System.Math.Sqrt(6.0 / (RowsAmount + ColumnsAmount));

        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                this[i, j] = random.NextSingle() * 2 * limit - limit;
            }
        }
    }

    public void InitializeHe()
    {
        float limit = (float)System.Math.Sqrt(6.0 / ColumnsAmount);

        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                this[i, j] = random.NextSingle() * 2 * limit - limit;
            }
        }
    }

    public float GetUnSquaredNorm()
    {
        float sum = 0;
        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                sum += values[i, j] * values[i, j];
            }
        }
        return sum;
    }

    public float GetNorm()
    {
        return (float)System.Math.Sqrt(GetUnSquaredNorm());
    }

    /// <summary>
    /// Gives access to the value at the given indexes
    /// </summary>
    /// <param name="i">Row index</param>
    /// <param name="j">Column index</param>
    /// <returns>Value at the given indexes</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when given indexes are out of range</exception>
    public float this[int i, int j]
    {
        get
        {
            if (i < 0 || i >= RowsAmount || j < 0 || j >= ColumnsAmount)
            {
                throw new IndexOutOfRangeException($"Given indexes ([{i},{j}]) are out of range for Matrix of size: {RowsAmount}x{ColumnsAmount}.");
            }
            return values[i, j];
        }
        set
        {
            if (i < 0 || i >= RowsAmount || j < 0 || j >= ColumnsAmount)
            {
                throw new IndexOutOfRangeException($"Given indexes ([{i},{j}]) are out of range for Matrix of size: {RowsAmount}x{ColumnsAmount}.");
            }
            values[i, j] = value;
        }
    }

    public IEnumerator<float> GetEnumerator()
    {
        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                yield return values[i, j];
            }
        }
    }

    public bool Equals(Matrix matrix)
    {
        if (matrix.RowsAmount != RowsAmount || matrix.ColumnsAmount != ColumnsAmount)
        {
            return false;
        }

        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                if (values[i, j] != matrix.values[i, j])
                {
                    return false;
                }
            }
        }

        return true;
    }

    public Matrix Copy()
    {
        return new Matrix((float[,])this.values.Clone());
    }

    public float[,] ToArray()
    {
        return (float[,])values.Clone();
    }

    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                sb.AppendFormat("{0,5} ", values[i, j].ToString("0.00"));
            }
            sb.Append("\n");
        }

        return sb.ToString();
    }

    public string ToFileString()
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < RowsAmount; i++)
        {
            for (int j = 0; j < ColumnsAmount; j++)
            {
                sb.Append(values[i, j].ToString() + " ");
            }
            sb.Append("\n");
        }

        return sb.ToString();
    }

    public static bool TryParse(string matrixString, out Matrix matrix)
    {
        string[] rows = matrixString.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        float[,] values = new float[rows.Length, rows[0].Split(' ', StringSplitOptions.RemoveEmptyEntries).Length];

        for (int i = 0; i < rows.Length; i++)
        {
            string[] columns = rows[i].Split(' ', StringSplitOptions.RemoveEmptyEntries);
            for (int j = 0; j < columns.Length; j++)
            {
                if(float.TryParse(columns[j], out float value) == false)
                {
                    matrix = new Matrix(0, 0);
                    return false;
                }
                if(values.GetLength(1) <= j || values.GetLength(0) <= i)
                {
                    matrix = new Matrix(0, 0);
                    return false;
                }
                values[i, j] = value;
            }
        }

        matrix = new Matrix(values);
        return true;
    }

    /// <summary>
    /// Creates new matrix which is result of dot product of two matrices
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>Result of dot product of two matrices</returns>
    public static Matrix DotProductMatrices(Matrix a, Matrix b)
    {
        if (a.ColumnsAmount != b.RowsAmount)
        {
            throw new ArgumentException("Number of columns in the first matrix must be equal to the number of rows in the second matrix");
        }

        Matrix tmp = new Matrix(a.RowsAmount, b.ColumnsAmount);
        for (int i = 0; i < a.RowsAmount; i++)
        {
            for (int j = 0; j < b.ColumnsAmount; j++)
            {
                for (int k = 0; k < a.ColumnsAmount; k++)
                {
                    tmp[i,j] += a[i, k] * b[k, j];
                }
            }
        }

        return tmp;
    }

    /// <summary>
    /// Creates new matrix which is result of element-wise multiplication of two matrices
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>Result of element-wise multiplication of two matrices</returns>
    public static Matrix ElementWiseMultiplyMatrices(Matrix a, Matrix b)
    {
        if (CheckIfDimensionsAreEqual(a, b) == false)
        {
            throw new ArgumentException("Matrices must have the same dimensions");
        }
        return EachElementAssignment(a, (i, j) => a.values[i, j] * b.values[i, j]);
    }

    /// <summary>
    /// Creates new matrix which is result of element-wise addition of two matrices
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>Result of element-wise addition of two matrices</returns>
    public static Matrix ElementWiseAddMatrices(Matrix a, Matrix b)
    {
        if (CheckIfDimensionsAreEqual(a, b) == false)
        {
            throw new ArgumentException("Matrices must have the same dimensions");
        }
        return EachElementAssignment(a, (i, j) => a.values[i, j] + b.values[i, j]);
    }

    /// <summary>
    /// Creates new matrix which is result of element-wise subtraction of two matrices
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>Result of element-wise subtraction of two matrices</returns>
    public static Matrix ElementWiseSubtractMatrices(Matrix a, Matrix b)
    {
        if (CheckIfDimensionsAreEqual(a, b) == false)
        {
            throw new ArgumentException("Matrices must have the same dimensions");
        }
        return EachElementAssignment(a, (i, j) => a.values[i, j] - b.values[i, j]);
    }

    public static Matrix operator *(Matrix a, float b)
    {
        return EachElementAssignment(a, (i, j) => a.values[i, j] * b);
    }

    public static Matrix operator +(Matrix a, float b)
    {
        return EachElementAssignment(a, (i, j) => a.values[i, j] + b);
    }

    /// <summary>
    /// Creates new matrix which is result of applying given function to each element of the a matrix.
    /// </summary>
    /// <param name="a">Matrix to apply function to</param>
    /// <param name="mathOperation">Function to apply</param>
    private static Matrix EachElementAssignment(Matrix a, Func<int, int, float> mathOperation)
    {
        Matrix result = new Matrix(a.RowsAmount, a.ColumnsAmount);

        for (int i = 0; i < a.RowsAmount; i++)
        {
            for (int j = 0; j < a.ColumnsAmount; j++)
            {
                result.values[i, j] = mathOperation(i, j);
            }
        }

        return result;
    }

    /// <summary>
    /// Checks if given matrices have the same dimensions
    /// </summary>
    /// <param name="a">First matrix</param>
    /// <param name="b">Second matrix</param>
    /// <returns>True if dimensions are the same, false otherwise</returns>
    private static bool CheckIfDimensionsAreEqual(Matrix a, Matrix b)
    {
        return a.RowsAmount == b.RowsAmount && a.ColumnsAmount == b.ColumnsAmount;
    }
}

