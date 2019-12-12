<h1>Neural Network in Java
</h1>

<p>This is a neural network implementation in Java. How do you use it?
Example:
</p>

<code>NeuralNetwork network = new NeuralNetwork(
      			new NetworkBuilder(4)
      				.setLayer(784, new ReluFunction())
      				.setLayer(100, new ReluFunction())
      				.setLayer(100, new ReluFunction())
      				.setLayer(10, new SoftmaxFunction())
      				.setCostFunction(new CrossEntropyCostFunction())
      				.setEvaluationFunction(new ArgMaxEvaluationFunction())
      				.setLearningRate(0.01)
      		);
</code>
