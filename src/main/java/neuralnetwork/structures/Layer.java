package neuralnetwork.structures;

import java.util.Arrays;
import math.ActivationFunction;
import math.TanhFunction;
import matrix.Matrix;

public class Layer {

	private int nodes;
	private Matrix bias;

	private double[] values;

	private boolean hasBias;

	private int indexInNetwork;

	private ActivationFunction function;

	Layer() {
		nodes = 0;
		bias = null;
		indexInNetwork = -1;
	}

	Layer(int nodes, Matrix bias, int indexInNetwork, boolean hasBias, ActivationFunction f) {
		this.nodes = nodes;
		this.bias = bias;
		this.indexInNetwork = indexInNetwork;
		this.hasBias = hasBias;
		this.function = f;
		values = new double[nodes];
	}

	Layer(int nodes, int indexInNetwork) {
		this(nodes, null, indexInNetwork, false, new TanhFunction());
		double[] in = new double[nodes];
		this.bias = Matrix.fromArray(in);
		values = new double[nodes];
	}

	Layer(int rows, Matrix biases, int indexInNetwork, boolean hasBias, double[] layer,
		ActivationFunction f) {
		this(rows, biases, indexInNetwork, hasBias, f);
		this.values = layer;
	}

	public Layer(int inputNodes, int i, ActivationFunction function) {
		this(inputNodes, null, i, false, function);
		double[] in = new double[inputNodes];
		this.bias = Matrix.fromArray(in);
		this.values = new double[inputNodes];
	}

	public Layer(int size, Matrix random, ActivationFunction function) {
		this.values = new double[size];
		this.nodes = size;
		this.bias = random;
		this.hasBias = true;
		this.function = function;
	}

	public void applyFunction() {
		Matrix prev = Matrix.fromArray(values);
		this.values = this.function.applyFunction(prev).toArray();
	}

	public void applyDerivative() {
		Matrix prev = Matrix.fromArray(values);
		this.values = this.function.applyDerivative(prev).toArray();
	}

	/**
	 * Calculate this layers new bias.
	 *
	 * @param delta the bias delta.
	 */
	public void calculateBias(Matrix delta) {
		this.bias.add(delta);
	}

	public void reset() {
		this.values = new double[nodes];
	}

	// -----------------
	// Getters and Setters
	// -----------------

	void setValue(double val, int index) {
		this.values[index] = val;
	}

	void setValue(Matrix values) {
		this.values = values.toArray();
	}

	public int getNodes() {
		return nodes;
	}

	public Matrix getBias() {
		return bias;
	}

	public void setBias(Matrix outputMatrix) {
		this.bias = outputMatrix;
	}

	public int getIndexInNetwork() {
		return indexInNetwork;
	}

	public int getNodesInLayer() {
		return this.nodes;
	}

	public Matrix getValues() {
		double[][] newValues = new double[this.values.length][1];
		int k = 0;
		for (double d : this.values) {
			newValues[k++][0] = d;
		}

		return new Matrix(newValues);
	}

	public void setValues(Matrix input) {
		this.values = input.toArray();
	}

	boolean hasBias() {
		return this.hasBias;
	}

	public ActivationFunction getFunction() {
		return this.function;
	}

	// -----------------------
	// END Getters and Setters
	// -----------------------


	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("Layer{");
		sb.append("bias=").append(bias);
		sb.append(", indexInNetwork=").append(indexInNetwork);
		sb.append(", nodes=").append(nodes);
		sb.append(", values=").append(Arrays.toString(values));
		sb.append('}');
		return sb.toString();
	}


}
