# Neural Network in Java

<p>This is a neural network implementation in Java. How do you use it?
Example:
</p>

## Example: 4 layer, MNIST evaluator

```java
ActivationFunction<SMatrix> f = new LeakyReluFunction<>(0.01);
ActivationFunction<SMatrix> softMax = new SoftmaxFunction<>();
LayeredNetworkBuilder<SMatrix> b = 
		new LayeredNetworkBuilder<SMatrix>(28 * 28) // Input size
		.layer(new NetworkLayer<>(f, 784, 0.1)) // First layer
		.layer(new NetworkLayer<>(f, 30, 0.1)) // Hidden 1
		.layer(new NetworkLayer<>(f, 70, 0.1)) // Hidden 2
		.layer(new NetworkLayer<>(softMax, 10)) // logits output
		.costFunction(new SmoothL1CostFunction<>()) // loss (here called cost)
		.evaluationFunction(new ArgMaxEvaluationFunction<>()) // pred class = real class
		.optimizer(new ADAM<>(0.01, 0.9, 0.999)) // ADAM optimizer
		.clipping(true) // norm clipping
		.initializer(
			new SimpleInitializer(MethodConstants.XAVIER,MethodConstants.SCALAR));

return b.create();
```

And then training the network with <code>train</code> like so:

```java
// Implement these yourself
List<NetworkInput> training = getDataFromDataSource("/foo/bar/myTrainingData");
List<NetworkInput> validation = getDataFromDataSource("/foo/bar/myValidationData");
int epochs = 100;
int batchSize = 32;

// Starts batch descent with the optimizer set in the constructor.
network.train(training, validation, epochs, batchSize);
```
	
