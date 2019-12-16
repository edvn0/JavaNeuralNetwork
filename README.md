<h1>Neural Network in Java
</h1>

<p>This is a neural network implementation in Java. How do you use it?
Example:
</p>
<h2>Example: 4 layer, MNIST evaluator</h2>

```java
NeuralNetwork network = new NeuralNetwork(
	new NetworkBuilder(4)
		.setFirstLayer(784)
		.setLayer(100, new ReluFunction())
		.setLayer(100, new ReluFunction())
		.setLastLayer(10, new SoftmaxFunction())
		.setCostFunction(new CrossEntropyCostFunction())
		.setEvaluationFunction(new ArgMaxEvaluationFunction())
		.setOptimizer(new ADAM(0.001, 0.9, 0.999))
);
```

And then training the network with <code>train</code> like so:

```java
List<NetworkInput> training = getDataFromDataSource("/foo/bar/myTrainingData");
List<NetworkInput> validation = getDataFromDataSource("/foo/bar/myValidationData");
int epochs = 100;
int batchSize = 32;

// Starts batch descent with the optimizer set in the constructor.
network.train(training, validation, epochs, batchSize);
```

<h2>The API</h2> 
After seeing how the example looks, you should also notice a couple of things which are easy to use:
<ul>
	<li>To classify or "predict" on new data, a simple call to the network's "predict(DenseMatrix input)" can be used.</li>
	<li>To output data into two separate graphs, one for the Loss and one for the Validation correctness, simply use "outputChart(String path)" method.</li>
	<li>To serialise and deserialise the network, use "writeObject" and "readObject"</li>
</ul>
	
