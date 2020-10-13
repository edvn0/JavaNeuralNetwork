package math.linearalgebra.ojalgo;

import static utilities.MathUtilities.simpleMap;

import java.util.Arrays;
import java.util.StringJoiner;
import java.util.function.Function;
import math.linearalgebra.Matrix;
import org.ojalgo.function.aggregator.Aggregator;
import org.ojalgo.matrix.Primitive64Matrix;
import utilities.MathUtilities;

public class OjAlgoMatrix implements Matrix<Primitive64Matrix> {

	private static final String NAME = "OjAlgo";
	protected Primitive64Matrix delegate;

	public OjAlgoMatrix(Primitive64Matrix in) {
		this.delegate = in;
	}

	public OjAlgoMatrix(double[] values) {
		this.delegate = Primitive64Matrix.FACTORY.columns(values);
	}

	public OjAlgoMatrix(double[][] data) {
		this.delegate = Primitive64Matrix.FACTORY.rows(data);
	}

	public OjAlgoMatrix(OjAlgoMatrix out) {
		this.delegate = out.delegate.copy().build();
	}

	@Override
	public String toString() {
		return new StringJoiner(", ", OjAlgoMatrix.class.getSimpleName() + "[", "]")
			.add("rawCopy=" + Arrays.deepToString(this.rawCopy()))
			.toString();
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}
		if (o == null || getClass() != o.getClass()) {
			return false;
		}
		OjAlgoMatrix matrix = (OjAlgoMatrix) o;
		return delegate.equals(matrix.delegate);
	}

	@Override
	public int hashCode() {
		return delegate.hashCode();
	}

	@Override
	public int rows() {
		return (int) this.delegate.countRows();
	}

	@Override
	public int cols() {
		return (int) this.delegate.countColumns();
	}

	@Override
	public OjAlgoMatrix mapElements(Function<Double, Double> mapping) {
		return new OjAlgoMatrix(simpleMap(mapping, this.delegate.toRawCopy2D()));
	}

	@Override
	public double sum() {
		return this.delegate.aggregateAll(Aggregator.SUM);
	}

	@Override
	public double max() {
		double[] array = this.delegate.toRawCopy1D();
		double best = Double.MIN_VALUE;
		for (int i = 0; i < array.length; i++) {
			if (array[i] > best) {
				best = array[i];
			}
		}
		return best;
	}

	@Override
	public int argMax() {
		double[] array = this.delegate.toRawCopy1D();
		return MathUtilities.argMax(array);
	}


	@Override
	public OjAlgoMatrix transpose() {
		return new OjAlgoMatrix(this.delegate.transpose());
	}

	@Override
	public Primitive64Matrix delegate() {
		return this.delegate;
	}

	@Override
	public OjAlgoMatrix divide(Matrix<Primitive64Matrix> right) {
		double[][] array = this.delegate.toRawCopy2D();
		double[][] other = right.delegate().toRawCopy2D();
		for (int i = 0; i < rows(); i++) {
			for (int j = 0; j < cols(); j++) {
				array[i][j] /= other[i][j];
			}
		}
		return new OjAlgoMatrix(array);
	}

	@Override
	public OjAlgoMatrix maxVector() {
		double max = this.max();

		double[][] values = new double[rows()][1];
		for (int i = 0; i < rows(); i++) {
			values[i][0] = max;
		}

		return new OjAlgoMatrix(values);
	}

	@Override
	public OjAlgoMatrix zeroes(int rows, int cols) {
		double[][] zeroes = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				zeroes[i][j] = 0;
			}
		}
		return new OjAlgoMatrix(Primitive64Matrix.FACTORY.rows(zeroes));
	}

	@Override
	public OjAlgoMatrix ones(int rows, int cols) {
		double[][] ones = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				ones[i][j] = 1;
			}
		}
		return new OjAlgoMatrix(Primitive64Matrix.FACTORY.rows(ones));
	}

	@Override
	public OjAlgoMatrix identity(int rows, int cols) {
		return new OjAlgoMatrix(Primitive64Matrix.FACTORY.makeEye(rows, cols));
	}

	@Override
	public double norm() {
		if (cols() != 1) {
			throw new IllegalArgumentException("Trying to take the norm of matrix... sus.");
		}

		return this.mapElements(e -> e * e).map(Matrix::sum);
	}

	@Override
	public OjAlgoMatrix hadamard(Matrix<Primitive64Matrix> otherMatrix) {
		double[][] elements = this.delegate.toRawCopy2D();
		double[][] out = otherMatrix.delegate().toRawCopy2D();
		for (int i = 0; i < elements.length; i++) {
			for (int j = 0; j < elements[0].length; j++) {
				out[i][j] *= (elements[i][j]);
			}
		}
		OjAlgoMatrix m = new OjAlgoMatrix(out);
		return m;
	}

	@Override
	public String name() {
		return NAME;
	}

	@Override
	public OjAlgoMatrix multiply(Matrix<Primitive64Matrix> otherMatrix) {
		return new OjAlgoMatrix(this.delegate.multiply(otherMatrix.delegate()));
	}

	@Override
	public OjAlgoMatrix multiply(double scalar) {
		return new OjAlgoMatrix(this.delegate.multiply(scalar));
	}

	@Override
	public OjAlgoMatrix add(Matrix<Primitive64Matrix> in) {
		return new OjAlgoMatrix(this.delegate.add(in.delegate()));
	}

	@Override
	public OjAlgoMatrix add(double in) {
		return new OjAlgoMatrix(this.delegate.add(in));
	}

	@Override
	public OjAlgoMatrix subtract(double in) {
		return new OjAlgoMatrix(this.delegate.subtract(in));
	}

	@Override
	public OjAlgoMatrix subtract(Matrix<Primitive64Matrix> in) {
		return new OjAlgoMatrix(this.delegate.subtract(in.delegate()));
	}

	@Override
	public OjAlgoMatrix divide(double in) {
		return new OjAlgoMatrix(this.delegate.divide(in));
	}

	@Override
	public double map(Function<Matrix<Primitive64Matrix>, Double> mapping) {
		return mapping.apply(this);
	}

	@Override
	public double[][] rawCopy() {
		return this.delegate.toRawCopy2D();
	}

	@Override
	public void setDelegate(Primitive64Matrix delegate) {
		this.delegate = delegate;
	}
}
