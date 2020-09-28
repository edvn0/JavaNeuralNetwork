package utilities.data;

import java.util.function.Function;
import neuralnetwork.inputs.NetworkInput;
import org.ujmp.core.DenseMatrix;

// TODO: Implement DataImport thingie
public class DataImport {

	private NetworkInput applyMinMax(String[] a) {
		double[][] data, label;
		data = new double[this.dataSize][1];
		label = new double[this.labelSize][1];

		for (int i = 0; i < dataSize; i++) {
			data[i][0] = minMaxNormalization(
				Double.parseDouble(a[i]), this.minMax[i][0], this.minMax[i][1]);
		}

		int oneHot = Integer.parseInt(a[a.length - 1]);
		label[oneHot][0] = 1;

		return new NetworkInput(DenseMatrix.Factory.importFromArray(data),
			DenseMatrix.Factory.importFromArray(label));
	}

	public enum DataMethod {

		MINMAX, Z_TRANSFORM, UNMODIFIED
	}

	private ImportMethod importMethod;

	boolean[] completed = new boolean[4];
	private int index;

	public Function<String[], NetworkInput> function;
	private int dataSize, labelSize;

	private double[][] minMax;
	private double[] means, standardDeviations;

	private DataMethod method;

	public DataImport() {
		index = 0;
	}

	public DataImport setMinMaxArray(double[][] minMax) {
		completed[index++] = true;
		if (minMax[0].length > 2) {
			throw new IllegalArgumentException(
				"Incorrect dimensions, only provide min and max for each row.");
		}
		this.minMax = minMax;
		return this;
	}

	public DataImport setMean(double[] mean) {
		this.completed[index++] = true;
		this.means = mean;
		return this;
	}

	public DataImport setStandardDeviation(double[] stddev) {
		this.completed[index++] = true;
		this.standardDeviations = stddev;
		return this;
	}

	public DataImport setDataSize(int k) {
		completed[index++] = true;
		this.dataSize = k;
		return this;
	}

	public DataImport setLabelSize(int k) {
		completed[index++] = true;
		this.labelSize = k;
		return this;
	}

	public DataImport setImportMethod(DataMethod m) {
		completed[index++] = true;
		this.method = m;
		return this;
	}

	public DataImport compile() {
		for (boolean b : completed) {
			if (!b) {
				throw new IllegalArgumentException(
					"The model is not yet ready for compilation.");
			}
		}

		switch (method) {
			case MINMAX:
				this.function = this::applyMinMax;
				break;
			case UNMODIFIED:
				this.function = this::applyUnmodified;
				break;
			case Z_TRANSFORM:
				this.function = this::applyZTransform;
				break;
			default:
				throw new RuntimeException("No method was provided, cannot compile the model.");
		}

		return this;
	}

	private NetworkInput applyUnmodified(final String[] strings) {
		double[][] data = new double[this.dataSize][1];
		double[][] label = new double[this.labelSize][1];

		for (int i = 0; i < strings.length - 1; i++) {
			data[i][0] = Double.parseDouble(strings[i]);
		}

		int index = Integer.parseInt(strings[strings.length - 1]);
		label[index][0] = 1;

		return new NetworkInput(DenseMatrix.Factory.importFromArray(data),
			DenseMatrix.Factory.importFromArray(label));
	}

	private NetworkInput applyZTransform(final String[] strings) {
		double[][] data = new double[this.dataSize][1];
		double[][] label = new double[this.labelSize][1];

		for (int i = 0; i < strings.length - 1; i++) {
			data[i][0] = zTransform(Double.parseDouble(strings[i]), i);
		}

		int index = Integer.parseInt(strings[strings.length - 1]);
		label[index][0] = 1;

		return new NetworkInput(DenseMatrix.Factory.importFromArray(data),
			DenseMatrix.Factory.importFromArray(label));
	}

	private double zTransform(final double parseDouble, int row) {
		return (parseDouble - means[row]) / (standardDeviations[row]);
	}

	public DataImport setMap(Function<String[], NetworkInput> e) {
		this.function = e;
		return this;
	}

	private static double minMaxNormalization(final double in, final double min,
		final double max) {
		return (in - min) / (max - min);
	}
}
