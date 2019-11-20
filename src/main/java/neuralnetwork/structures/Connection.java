package neuralnetwork.structures;


import matrix.Matrix;

public class Connection {

	private Layer from, to;
	private Matrix weights;

	Connection() {
		from = null;
		to = null;
	}

	Connection(Layer from, Layer to) {
		this.from = from;
		this.to = to;
	}

	Connection(Layer from, Layer to, Matrix weights) {
		this(from, to);
		this.weights = weights;
	}

	public Matrix getWeights() {
		return this.weights;
	}

	public Layer getFrom() {
		return this.from;
	}

	public Layer getTo() {
		return this.to;
	}

	@Override
	public String toString() {
		return "[" + this.from.toString() + "," + this.to.toString() + "]";
	}
}
