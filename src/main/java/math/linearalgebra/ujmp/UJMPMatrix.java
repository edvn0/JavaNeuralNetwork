package math.linearalgebra.ujmp;

import static utilities.MathUtilities.simpleMap;

import java.util.Arrays;
import java.util.StringJoiner;
import java.util.function.Function;

import org.ujmp.core.calculation.Calculation.Ret;

import math.linearalgebra.Matrix;
import utilities.MathUtilities;
import utilities.exceptions.MatrixException;

public class UJMPMatrix implements Matrix<org.ujmp.core.Matrix> {

	private static final String NAME = "UJMPMatrix";
	private org.ujmp.core.Matrix delegate;

	public UJMPMatrix(org.ujmp.core.Matrix in) {
		this.delegate = in;
	}

	public UJMPMatrix(double[] values) {
		double[][] transpose = new double[values.length][1];
		int i = 0;
		for (var d : values) {
			transpose[i++] = new double[]{d};
		} // Hack, need to transpose, because UJMP thinks everyone just LOOOVEs row
		// vectors.
		this.delegate = org.ujmp.core.Matrix.Factory.importFromArray(transpose);
	}

	public UJMPMatrix(double[][] data) {
		this.delegate = org.ujmp.core.Matrix.Factory.importFromArray(data);
	}

	public UJMPMatrix(UJMPMatrix out) {
		this.delegate = out.delegate;
	}

	@Override
	public int rows() {
		return (int) this.delegate.getRowCount();
	}

	@Override
	public int cols() {
		return (int) this.delegate.getColumnCount();
	}

	@Override
	public UJMPMatrix multiply(Matrix<org.ujmp.core.Matrix> otherMatrix) {
		return new UJMPMatrix(this.delegate.mtimes(otherMatrix.delegate()));
	}

	@Override
	public UJMPMatrix multiply(double scalar) {
		return new UJMPMatrix(this.delegate.times(scalar));
	}

	@Override
	public UJMPMatrix add(Matrix<org.ujmp.core.Matrix> in) {
		return new UJMPMatrix(this.delegate.plus(in.delegate()));
	}

	@Override
	public UJMPMatrix add(double in) {
		return new UJMPMatrix(this.delegate.plus(in));
	}

	@Override
	public UJMPMatrix subtract(double in) {
		return new UJMPMatrix(this.delegate.minus(in));
	}

	@Override
	public UJMPMatrix subtract(Matrix<org.ujmp.core.Matrix> in) {
		return new UJMPMatrix(this.delegate.minus(in.delegate()));
	}

	@Override
	public UJMPMatrix divide(double in) {
		return new UJMPMatrix(this.delegate.divide(in));
	}

	@Override
	public double map(Function<Matrix<org.ujmp.core.Matrix>, Double> mapping) {
		return mapping.apply(this);
	}

	@Override
	public UJMPMatrix mapElements(Function<Double, Double> mapping) {

		return new UJMPMatrix(simpleMap(mapping, this.delegate.toDoubleArray()));
	}

	@Override
	public String toString() {
		return new StringJoiner(", ", UJMPMatrix.class.getSimpleName() + "[", "]")
			.add("rawCopy=" + Arrays.deepToString(this.rawCopy()))
			.toString();
	}

	@Override
	public double sum() {
		return this.delegate.getValueSum();
	}

	@Override
	public double max() {
		return this.delegate.max(org.ujmp.core.calculation.Calculation.Ret.NEW, 0).doubleValue();
	}

	@Override
	public UJMPMatrix transpose() {
		return new UJMPMatrix(this.delegate.transpose());
	}

	@Override
	public org.ujmp.core.Matrix delegate() {
		return this.delegate;
	}

	@Override
	public UJMPMatrix divide(Matrix<org.ujmp.core.Matrix> right) {
		return new UJMPMatrix(this.delegate.divide(right.delegate()));
	}

	@Override
	public UJMPMatrix maxVector() {
		double max = this.max();

		double[][] vector = new double[rows()][1];
		for (int i = 0; i < rows(); i++) {
			vector[i][0] = max;
		}

		return new UJMPMatrix(vector);
	}

	@Override
	public UJMPMatrix zeroes(int rows, int cols) {
		return new UJMPMatrix(org.ujmp.core.Matrix.Factory.zeros(rows, cols));
	}

	@Override
	public UJMPMatrix ones(int rows, int cols) {
		return new UJMPMatrix(org.ujmp.core.Matrix.Factory.ones(rows, cols));
	}

	@Override
	public UJMPMatrix identity(int rows, int cols) {
		return new UJMPMatrix(org.ujmp.core.Matrix.Factory.eye(rows, cols));
	}

	@Override
	public int argMax() {
		double[] array = this.delegate.toDoubleArray()[0];
		return MathUtilities.argMax(array);

	}

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}
		if (o == null || getClass() != o.getClass()) {
			return false;
		}
		UJMPMatrix matrix = (UJMPMatrix) o;
		return delegate.equals(matrix.delegate);
	}

	@Override
	public int hashCode() {
		return delegate.hashCode();
	}

	@Override
	public UJMPMatrix hadamard(Matrix<org.ujmp.core.Matrix> otherMatrix) {
		return new UJMPMatrix(this.delegate.times(otherMatrix.delegate()));
	}

	@Override
	public double norm() {

		if (cols() != 1) {
			throw new MatrixException("Not a vector.");
		}

		return this.delegate.times(this.delegate).sqrt(Ret.NEW).getValueSum();
	}

	@Override
	public String name() {
		return NAME;
	}

	@Override
	public double[][] rawCopy() {
		return this.delegate.toDoubleArray();
	}

	@Override
	public void setDelegate(org.ujmp.core.Matrix delegate) {
		this.delegate = delegate;
	}
}
