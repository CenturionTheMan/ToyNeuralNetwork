using ToyNeuralNetwork.Math;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ToyNeuralNetwork.ImageProcessing;

public static class ImageEditor
{
    private static Random random = new Random();

    public static Matrix? CutOffBorderToSquare(this Matrix matrix, (float min, float max) contentColor, int padding, float bgColor = 1.0f)
    {
        int top = int.MinValue;
        int bottom = int.MaxValue;
        int left = int.MaxValue;
        int right = int.MinValue;

        //bottom
        for (int y = 0; y < matrix.RowsAmount; y++)
        {
            for (int x = 0; x < matrix.ColumnsAmount; x++)
            {
                if(matrix[y,x] < contentColor.min || matrix[y,x] > contentColor.max) continue;

                if(y < bottom)
                {
                    bottom = y;
                    break;
                }
            }
        }

        //top
        for (int y = matrix.RowsAmount - 1; y >= 0; y--)
        {
            for (int x = 0; x < matrix.ColumnsAmount; x++)
            {
                if(matrix[y,x] < contentColor.min || matrix[y,x] > contentColor.max) continue;

                if(y > top)
                {
                    top = y;
                    break;
                }
            }
        }

        //right
        for (int x = matrix.ColumnsAmount - 1; x >= 0; x--)
        {
            for (int y = 0; y < matrix.RowsAmount; y++)
            {
                if(matrix[y,x] < contentColor.min || matrix[y,x] > contentColor.max) continue;

                if(x > right)
                {
                    right = x;
                    break;
                }
            }
        }

        //left
        for (int x = 0; x < matrix.ColumnsAmount; x++)
        {
            for (int y = 0; y < matrix.RowsAmount; y++)
            {
                if(matrix[y,x] < contentColor.min || matrix[y,x] > contentColor.max) continue;

                if(x < left)
                {
                    left = x;
                    break;
                }
            }
        }
    
        if(top == int.MinValue || bottom == int.MaxValue || left == int.MaxValue || right == int.MinValue)
            return null;

        int size = System.Math.Max(right - left + 1, top - bottom + 1) + padding *2;
        int offset = System.Math.Min(left, bottom) - padding;

        Matrix result = new Matrix(size, size);

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int oldY = offset + y;
                int oldX = offset + x;
                result[y, x] = (oldY < matrix.RowsAmount && oldX < matrix.ColumnsAmount)? matrix[oldY, oldX] : bgColor;
            }
        }

        return result;
    }

    public static Matrix ToStrictBlackWhite(this Matrix matrix, float threshold = 0.5f)
    {
        return matrix.ApplyFunction(x => x > threshold ? 1 : 0);
    }

    public static Matrix RandomShiftMatrixImage(this Matrix matrix, float minScale, float maxScale, float minRotationDeg, float maxRotationDeg, float bgColor)
    {
        float scale = random.NextSingle() * (maxScale - minScale) + minScale;
        float rotation = random.NextSingle() * (maxRotationDeg - minRotationDeg) + minRotationDeg;

        Matrix result = matrix.Rotate(rotation);

        if (scale != 1.0f)
            result = scale < 1.0f ? result.Scale(scale, bgColor).AddPadding(matrix.ColumnsAmount, matrix.RowsAmount, bgColor) : result.ScaleUpWithPreservedDimensions(scale, bgColor);

        return result;
    }

    public static Matrix Rotate(this Matrix matrix, float angleDeg)
    {
        int rows = matrix.RowsAmount;
        int cols = matrix.ColumnsAmount;

        Matrix rotatedMatrix = new Matrix(rows, cols);

        float angleRad = -angleDeg * (float)System.Math.PI / 180.0f;
        float cos = (float)System.Math.Cos(angleRad);
        float sin = (float)System.Math.Sin(angleRad);

        float centerX = cols / 2.0f;
        float centerY = rows / 2.0f;

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                float srcXf = (x - centerX) * cos + (y - centerY) * sin + centerX;
                float srcYf = -(x - centerX) * sin + (y - centerY) * cos + centerY;

                int srcX1 = (int)System.Math.Floor(srcXf);
                int srcX2 = srcX1 + 1;
                int srcY1 = (int)System.Math.Floor(srcYf);
                int srcY2 = srcY1 + 1;

                srcX1 = System.Math.Max(0, System.Math.Min(cols - 1, srcX1));
                srcX2 = System.Math.Max(0, System.Math.Min(cols - 1, srcX2));
                srcY1 = System.Math.Max(0, System.Math.Min(rows - 1, srcY1));
                srcY2 = System.Math.Max(0, System.Math.Min(rows - 1, srcY2));

                float weightX2 = srcXf - srcX1;
                float weightX1 = 1.0f - weightX2;
                float weightY2 = srcYf - srcY1;
                float weightY1 = 1.0f - weightY2;

                float interpolatedValue = weightX1 * (weightY1 * matrix[srcX1, srcY1] + weightY2 * matrix[srcX1, srcY2])
                                        + weightX2 * (weightY1 * matrix[srcX2, srcY1] + weightY2 * matrix[srcX2, srcY2]);

                rotatedMatrix[x, y] = interpolatedValue;
            }
        }

        return rotatedMatrix;
    }

    public static Matrix AddPadding(this Matrix matrix, int desiredWidth, int desiredHeight, float paddingColor)
    {
        int originalRows = matrix.RowsAmount;
        int originalCols = matrix.ColumnsAmount;

        Matrix paddedMatrix = new Matrix(desiredHeight, desiredWidth);
        paddedMatrix = paddedMatrix.ApplyFunction(x => paddingColor);

        int startX = (desiredWidth - originalCols) / 2;
        int startY = (desiredHeight - originalRows) / 2;

        for (int y = 0; y < originalRows; y++)
        {
            for (int x = 0; x < originalCols; x++)
            {
                int newY = y + startY;
                int newX = x + startX;

                if (newX >= 0 && newX < desiredWidth && newY >= 0 && newY < desiredHeight)
                {
                    paddedMatrix[newY, newX] = matrix[y, x];
                }
            }
        }

        return paddedMatrix;
    }

    public static Matrix ResizeSquare(this Matrix matrix, int desiredSize, float bgColor)
    {
        if(matrix.RowsAmount != matrix.ColumnsAmount)
            throw new ArgumentException("Matrix must be square", nameof(matrix));

        float scale = (float)desiredSize / matrix.RowsAmount;
        var mat = matrix.Scale(scale, bgColor);

        if(mat.RowsAmount < desiredSize || mat.ColumnsAmount < desiredSize)
        {
            mat = mat.AddPadding(desiredSize, desiredSize, bgColor);
        }

        return mat;
    }

    public static Matrix Scale(this Matrix matrix, float scale, float bgColor)
    {
        int rows = matrix.RowsAmount;
        int cols = matrix.ColumnsAmount;

        int scaledWidth = (int)(cols * scale);
        int scaledHeight = (int)(rows * scale);

        Matrix scaledMatrix = new Matrix(scaledHeight, scaledWidth);

        float invScale = 1.0f / scale;

        float offsetX = (cols - 1) / 2.0f - (scaledWidth - 1) / (2.0f * scale);
        float offsetY = (rows - 1) / 2.0f - (scaledHeight - 1) / (2.0f * scale);

        for (int y = 0; y < scaledHeight; y++)
        {
            for (int x = 0; x < scaledWidth; x++)
            {
                float srcXf = x * invScale + offsetX;
                float srcYf = y * invScale + offsetY;

                int srcX1 = (int)System.Math.Floor(srcXf);
                int srcX2 = System.Math.Min(srcX1 + 1, cols - 1);
                int srcY1 = (int)System.Math.Floor(srcYf);
                int srcY2 = System.Math.Min(srcY1 + 1, rows - 1);

                if (srcX1 >= 0 && srcX2 >= 0 && srcY1 >= 0 && srcY2 >= 0 && srcX1 < cols && srcX2 < cols && srcY1 < rows && srcY2 < rows)
                {
                    float weightX2 = srcXf - srcX1;
                    float weightX1 = 1.0f - weightX2;
                    float weightY2 = srcYf - srcY1;
                    float weightY1 = 1.0f - weightY2;

                    float interpolatedValue = weightX1 * (weightY1 * matrix[srcY1, srcX1] + weightY2 * matrix[srcY2, srcX1])
                                            + weightX2 * (weightY1 * matrix[srcY1, srcX2] + weightY2 * matrix[srcY2, srcX2]);

                    scaledMatrix[y, x] = interpolatedValue;
                }
                else
                {
                    scaledMatrix[y, x] = bgColor;
                }
            }
        }

        return scaledMatrix;
    }

    public static Matrix ScaleUpWithPreservedDimensions(this Matrix matrix, float scale, float bgColor)
    {
        if (scale < 1.0f)
            throw new ArgumentException("Scale factor must be greater than 1.0", nameof(scale));

        int rows = matrix.RowsAmount;
        int cols = matrix.ColumnsAmount;

        Matrix scaledMatrix = new Matrix(rows, cols);

        float invScale = 1.0f / scale;

        int scaledWidth = (int)(cols * scale);
        int scaledHeight = (int)(rows * scale);

        float offsetX = (cols - scaledWidth) / 2.0f;
        float offsetY = (rows - scaledHeight) / 2.0f;

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                float srcXf = (x - offsetX) * invScale;
                float srcYf = (y - offsetY) * invScale;

                if (srcXf >= 0 && srcXf < scaledWidth - 1 && srcYf >= 0 && srcYf < scaledHeight - 1)
                {
                    int srcX1 = (int)System.Math.Floor(srcXf);
                    int srcX2 = srcX1 + 1;
                    int srcY1 = (int)System.Math.Floor(srcYf);
                    int srcY2 = srcY1 + 1;

                    float weightX2 = srcXf - srcX1;
                    float weightX1 = 1.0f - weightX2;
                    float weightY2 = srcYf - srcY1;
                    float weightY1 = 1.0f - weightY2;

                    float interpolatedValue = weightX1 * (weightY1 * matrix[srcY1, srcX1] + weightY2 * matrix[srcY2, srcX1])
                                            + weightX2 * (weightY1 * matrix[srcY1, srcX2] + weightY2 * matrix[srcY2, srcX2]);

                    scaledMatrix[y, x] = interpolatedValue;
                }
                else
                {
                    scaledMatrix[y, x] = bgColor;
                }
            }
        }

        return scaledMatrix;
    }


    public static Matrix LoadFromImage(string path)
    {
        using var image = Image.Load<Rgba32>(path);
        Matrix matrix = new Matrix(image.Height, image.Width);

        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                Rgba32 pixel = image[x, y];
                float value = (pixel.R + pixel.G + pixel.B) / 3.0f / 255.0f;
                matrix[y, x] = value;
            }
        }

        return matrix;
    }

    public static bool SaveAsJpeg(this Matrix matrix, string path)
    {
        int height = matrix.RowsAmount;
        int width = matrix.ColumnsAmount;

        using var image = new Image<Rgba32>(width, height);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte value = (byte)(matrix[y, x] * 255);
                image[x, y] = new Rgba32(value, value, value, 255);
            }
        }

        try
        {
            image.SaveAsJpeg(path);
            return true;
        }
        catch
        {
            return false;
        }
    }


    public static bool SaveAsPng(this Matrix matrix, string path)
    {
        int height = matrix.RowsAmount;
        int width = matrix.ColumnsAmount;

        using var image = new Image<Rgba32>(width, height);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte value = (byte)(matrix[y, x] * 255);
                image[x, y] = new Rgba32(value, value, value, 255); 
            }
        }

        try
        {
            image.SaveAsPng(path);
            return true;
        }
        catch
        {
            return false;
        }
    }

}