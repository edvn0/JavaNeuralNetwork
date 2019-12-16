<h1>Neural Network in Java
</h1>

<p>This is a neural network implementation in Java. How do you use it?
Example:
</p>
<h2>Example: 4 layer, MNIST evaluator</h2>


'''java
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
'''
