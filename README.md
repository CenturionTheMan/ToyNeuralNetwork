[![NuGet](https://img.shields.io/nuget/v/ToyNeuralNetwork.svg)](https://www.nuget.org/packages/ToyNeuralNetwork/#readme-body-tab)

# Introduction

The main motivation for the project was to create a library that enables interaction with neural networks through a high-level interface. The aim was to provide the user with a set of specific classes and methods that manage the library's lower-level relationships in an accessible manner.

The tool was designed with the capability to handle any data that can be represented as a set of matrices. However, for testing purposes, the "Quick Draw Dataset" was used, and as a result, the library was enriched with a module specifically for handling this dataset.

For implementation purposes, the C# programming language was chosen in combination with the .NET development platform.

# Table of contents

- [Introduction](#introduction)
- [Table of contents](#table-of-contents)
- [Scope of functionalities](#scope-of-functionalities)
- [Library structure](#library-structure)
  - [Mathematical Module](#mathematical-module)
  - [Image Processing Module](#image-processing-module)
  - [Utils Module](#utils-module)
  - ["Quick, Draw!" Data Handling Module](#quick-draw-data-handling-module)
  - [Neural Networks Module](#neural-networks-module)
- [Developer guide](#developer-guide)

# Scope of functionalities

The library supports the creation and management of both multilayer perceptron (MLP) networks and convolutional networks.

Support for the following layer types has been implemented:

- **Fully connected layer** - supported parameters: number of neurons, activation function,
- **Dropout layer** - supported parameters: dropout rate,
- **Convolutional layer** - supported parameters: filter size, depth, activation function,
- **Max pooling layer** - supported parameters: window size, stride.

Additionally, the library uses a reshape layer when transitioning between feature extraction layers and classification layers. However, this layer is automatically generated by the library during the creation of a new network instance.

The library also implements support for three activation functions:

- ReLU
- Sigmoid
- Softmax

It is important to note that the library only supports creating networks where the final layer is a fully connected layer with a softmax activation function.  
The network error is calculated using the cross-entropy cost function.

# Library structure

![](.github/README_IMG/lib_all_diagram.png?raw=true)

In Figure, it can be observed that the system of dependencies and relationships within the library is relatively complex. However, the classes it contains can be divided into five modules:

- Mathematical Module
- Image Processing Module
- Utils Module
- "Quick, Draw!" Data Handling Module
- Neural Networks Module

## Mathematical Module

![](.github/README_IMG/math.png?raw=true)

The **Mathematical Module** is the core of the entire library. It implements a matrix class along with a set of mathematical operations that can be performed on matrices.

The main class representing the structure is `Matrix`, which is based on a two-dimensional array of floating-point numbers that stores the values. Along with the `MatrixExtender` and `ActivationFunctionHandler` classes, the library provides a complete set of operations on matrices, including:

- Addition, subtraction, multiplication, and division of matrices by a constant value,
- Element-wise addition, subtraction, multiplication, and division of matrices,
- Scalar product,
- Transposition,
- Convolution operation,
- Sum of values,
- Maximum value of a matrix,
- Rotation of a matrix by 180 degrees,
- Maximum pooling aggregation,
- Applying selected activation functions,
- Applying arbitrary functions with a specified formula, and much more.

Additionally, the module includes a `Statistics` class, which contains a method for calculating the slope and the intercept of a linear regression line for given points.

The library utilizes this method during the training process in the `Trainer` class.

The final class is `MatrixHelper`, which contains methods for calculating corresponding indices between a two-dimensional array and a one-dimensional array.

## Image Processing Module

![](.github/README_IMG/imageProcessing.png?raw=true)

The **Image Processing Module** consists of only one class: `ImageEditor`. In this case, the matrix represents a single channel of an image, and the available set of functions is used for editing it. The implemented methods are used in the library to modify the input data to better crop the samples and introduce noise in the form of randomly modified rotations and sizes.

Additionally, the library allows for loading an image into a matrix and saving the structure as a PNG or JPEG file. The `SixLabors.ImageSharp` library is used for these operations.

## Utils Module

![](.github/README_IMG/utils.png?raw=true)

The **Utils Module** contains declarations of enumerations and tools for basic file management.

The available methods allow for verifying the existence of a file or folder and enable changing the file extension in the path.

Additionally, support for XML and CSV files has been implemented. XML files are handled using the `System.Xml` namespace for saving and reading user-created neural networks. CSV files are used to save information about the progress during network training (discussed in more detail in the **Neural Networks Module**).

## "Quick, Draw!" Data Handling Module

![](.github/README_IMG/quickDrawHandler.png?raw=true)

The **"Quick, Draw!" Data Handling Module** contains classes responsible for transforming the available data from the "Quick Draw" dataset in .npy format to the format used by the library.

The first structure to be discussed is the `QuickDrawSample`. A single instance of it represents a single file of samples from the dataset (of the same class). It has two fields:

- `category`, which indicates the category of the samples contained in it,
- `data`, which is a set of matrices representing the samples from the file.

The next class is `QuickDrawSet`, which represents a dataset ready for training. The methods contained in it are used for proportionally splitting the data into training and testing sets.

Additionally, for the nine classes implemented in the presentation layer, `QuickDrawSet` allows for transforming data labels into numerical values (and vice versa).

## Neural Networks Module

![](.github/README_IMG/neuralNetwork.png?raw=true)

Finally, the most complex of the modules, the **Neural Networks Module**. Within it, we can distinguish one subgroup: **Layers**. This contains all the layers available to the user. These layers implement the `ILayer` interface, which declares methods for forwarding and backwarding data during the training process, as well as methods for saving the layer to an XML file.

The most important class of the module (and of the entire library) is `NeuralNetwork`. Using it, the user creates a selected model and then utilizes it. This class allows the construction of both a multilayer perceptron (MLP) network and a convolutional network. Whether the model is a convolutional network or MLP is defined by the layers selected by the user when creating a new instance. The class provides a range of functionalities, including:

- Methods to initialize network training,
- Events triggered during training progress,
- Methods to return network predictions given a specific input,
- Methods for saving and reading the network from an XML file.

The constructors of the `NeuralNetwork` class require information about the shape of the data and the objects of the `LayerTemplate` class. This class contains minimal business logic, and its main task is to store data upon which the actual layers implementing the `ILayer` interface are created. The constructor of the class is private, so its instances are created using static methods, which immediately specify the set of parameters needed to create a given layer.

The final class is `Trainer`, which extends the configuration capabilities of the network training process. The `Trainer` enhances the network training by adding two aspects:

- Automatically adjusting the learning rate parameter,
- Saving information about the training process to a file.

The automatic adjustment of the learning rate parameter is intended to reduce its value when the network stops making progress during training. In practice, the trainer takes a few of the most recent batch errors (the number of which is dependent on the `patience` parameter, where a value of 1 means all batches in an epoch) and calculates the slope of the regression line based on them. If the slope is non-negative, it indicates that the network error is not decreasing, so the trainer will reduce the learning rate.

During training, the class can log the progress of the learning process and, at the end of training, save to files:

- The error for each sample in each iteration,
- The error for each batch in each iteration,
- The average, maximum, minimum error, and network accuracy for each epoch,
- The final accuracy for each class individually,
- Trainer configurations,
- The trained network.

# Developer guide

Using the library involves several steps. The first step is to create an array of `LayerTemplate` objects.

```csharp
LayerTemplate[] templates = [
    LayerTemplate.CreateConvolutionLayer(kernelSize: 5, depth: 12, activationFunction: ActivationFunction.ReLU),
    LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
    LayerTemplate.CreateConvolutionLayer(kernelSize: 3, depth: 24, activationFunction: ActivationFunction.Sigmoid),
    LayerTemplate.CreateMaxPoolingLayer(poolSize: 2, stride: 2),
    LayerTemplate.CreateDropoutLayer(dropoutRate: 0.5f),
    LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
    LayerTemplate.CreateFullyConnectedLayer(layerSize: 64, activationFunction: ActivationFunction.ReLU),
    LayerTemplate.CreateFullyConnectedLayer(layerSize: 9, activationFunction: ActivationFunction.Softmax)
];
```

Next, create an instance of the `NeuralNetwork` class.

```csharp
var nn = new NeuralNetwork(inputDepth: 1, inputRowsAmount: 28, inputColumnsAmount: 28, layerTemplates: templates);
```

The `inputDepth` parameter represents the number of image channels (in the above example, it is 1, meaning the image is black and white). The `inputRowsAmount` and `inputColumnsAmount` parameters represent the height and width of the image, respectively. Finally, `layerTemplates` is the network architecture in the form of a collection of `LayerTemplate` objects.

Once the network object is created, it can be trained using the `Train` or `TrainOnNewTask` methods. The first works on the current thread, while the second operates on a `Task` construct, which is a thread wrapper in C#.

```csharp
// Declaration of a sample dataset (a single sample filled with zeros)
(Matrix[], Matrix)[] data = [([new Matrix(28, 28)], new Matrix(9, 1))];

// Calling the method to start training
nn.Train(data: data, learningRate: 0.01f, epochAmount: 30, batchSize: 50);
```

The network can also be trained using the `Trainer` class. This class allows saving the training process to files and using the functionality for automatically lowering the learning rate parameter value - Patience.

```csharp
(Matrix[], Matrix)[] testData = [([new Matrix(28, 28)], new Matrix(9, 1))];

var trainer = new Trainer(neuralNetwork: nn, data: data, initialLearningRate: 0.01f, minLearningRate: 0.001f, epochAmount: 30, batchSize: 50);

(Task task, CancellationTokenSource cts) = trainer.SetPatience(initialIgnore: 0.5f, patience: 0.3f, learningRateModifier: (lr, epoch) => lr - 0.001f)
                                                  .SetLogSaving(outputDirPath: "dirPath", saveNN: true, testData: testData, out string trainingLogDirectory)
                                                  .RunTrainingOnTask();
```

When creating a `Trainer` object, the minimum value for the learning rate parameter must be provided separately. If the network reaches this value during training, the training will terminate prematurely - at the end of the current epoch. The `SetPatience` method specifies that this functionality should be included during training. The `initialIgnore` parameter delays the start of this functionality. For the value used in the example, the verification of the decline in the average error will start halfway through the first epoch. The `patience` variable defines how many (and which) batches will be checked to determine whether the network is making progress. For a value of 0.3, the regression line will be calculated for (approximately) 1/3 of the batches in an epoch.

The `SetLogSaving` function determines whether the progress of the training process should be saved. Its parameters are:

- `outputDirPath` - the path to the folder where logs will be saved. A new folder will be created within it, named after the day and time the training started.
- `saveNN` - specifies whether the trained network should also be saved.
- `testData` - the samples used to calculate the network's effectiveness.
- `trainingLogDirectory` - the path to the folder where the log files are saved.

Finally, `RunTrainingOnTask` starts the training on a new thread. This method returns a `Task` object representing the ongoing task, and a `CancellationTokenSource`, which allows the training process to be stopped.

Once the network is trained, we can use it via the provided methods:

```csharp
// Example of using the network to predict the class for a new sample
Matrix prediction = nn.Predict([new Matrix(28, 28)]);

// Example of calculating the correctness of the network
float correctness = nn.CalculateCorrectness(testData: testData);

// Example of saving the network to an XML file
nn.SaveToXmlFile(path: "savingPath", testCorrectness: correctness);
```

The following listing shows examples of how to use the trained network. The main functionalities include:

- **Predict** - returns a matrix with predicted values for the given sample.
- **CalculateCorrectness** - calculates the network's correctness using the provided test set.
- **SaveToXmlFile** - saves the network to an XML file. The `testCorrectness` parameter is optional and allows saving the network’s correctness information.
