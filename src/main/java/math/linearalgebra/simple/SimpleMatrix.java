package math.linearalgebra.simple;

import java.util.function.Function;
import math.linearalgebra.Matrix;
import utilities.MathUtilities;
import utilities.exceptions.MatrixException;

public class SimpleMatrix implements Matrix<SMatrix> {

	private SMatrix delegate;

	public SimpleMatrix(SMatrix times) {
		this.delegate = times;
	}

	public SimpleMatrix(double[][] vals) {
		this.delegate = new SMatrix(vals);
	}

	public SimpleMatrix(SimpleMatrix out) {
		this.delegate = out.delegate;
	}

	public SimpleMatrix(double[] ds) {
		this.delegate = new SMatrix(ds);
	}

	@Override
	public int rows() {
		return delegate.rows();
	}

	@Override
	public int cols() {
		return delegate.cols();
	}

	@Override
	public SimpleMatrix multiply(Matrix<SMatrix> otherMatrix) {
		return new SimpleMatrix(this.delegate.times(otherMatrix.delegate()));
	}

	@Override
	public SimpleMatrix hadamard(Matrix<SMatrix> otherMatrix) {
		return new SimpleMatrix(this.delegate.hadamard(otherMatrix.delegate()));
	}

	@Override
	public SimpleMatrix multiply(double scalar) {
		return new SimpleMatrix(this.delegate.times(scalar));
	}

	@Override
	public SimpleMatrix add(Matrix<SMatrix> in) {
		return new SimpleMatrix(this.delegate.plus(in.delegate()));
	}

	@Override
	public SimpleMatrix add(double in) {
		return new SimpleMatrix(this.delegate.plus(in));
	}

	@Override
	public SimpleMatrix subtract(double in) {
		return new SimpleMatrix(this.delegate.minus(in));
	}

	@Override
	public SimpleMatrix subtract(Matrix<SMatrix> in) {
		return new SimpleMatrix(this.delegate.minus(in.delegate()));
	}

	@Override
	public SimpleMatrix divide(double in) {
		return new SimpleMatrix(this.delegate.divide(in));
	}

	@Override
	public double map(Function<Matrix<SMatrix>, Double> mapping) {
		return mapping.apply(this);
	}

	@Override
	public SimpleMatrix mapElements(Function<Double, Double> mapping) {
		return new SimpleMatrix(MathUtilities.simpleMap(mapping, this.rawCopy()));
	}

	@Override
	public double sum() {
		return this.delegate.sum();
	}

	@Override
	public double max() {
		return this.delegate.max();
	}

	@Override
	public int argMax() {
		return this.delegate.argMax();
	}

	@Override
	public SMatrix delegate() {
		return delegate;
	}

	@Override
	public void setDelegate(SMatrix delegate) {
		this.delegate = delegate;
	}

	@Override
	public SimpleMatrix transpose() {
		return new SimpleMatrix(this.delegate.transpose());
	}

	@Override
	public SimpleMatrix divide(Matrix<SMatrix> right) {
		return new SimpleMatrix(this.delegate.divide(right.delegate()));
	}

	@Override
	public SimpleMatrix maxVector() {
		return new SimpleMatrix(this.delegate.maxVector());
	}

	@Override
	public SimpleMatrix zeroes(int rows, int cols) {
		return new SimpleMatrix(new double[rows][cols]);
	}

	@Override
	public SimpleMatrix ones(int rows, int cols) {
		double[][] ones = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				ones[i][j] = 1d;
			}
		}
		return new SimpleMatrix(ones);
	}

	@Override
	public SimpleMatrix identity(int rows, int cols) {
		double[][] id = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (i == j) {
					id[i][j] = 1d;
				}
			}
		}
		return new SimpleMatrix(id);
	}

	@Override
	public double norm() throws MatrixException {
		return this.delegate.norm();
	}

	@Override
	public String name() {
		return "SimpleMatrix";
	}

	@Override
	public double[][] rawCopy() {
		return this.delegate.rawCopy();
	}

	@Override
	public Matrix<SMatrix> copy() {
		return new SimpleMatrix(this.delegate.rawCopy());
	}

	@Override
	public String toString() {
		return "SimpleMatrix=[" + this.delegate.toString() + "]";
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}
		if (o == null || getClass() != o.getClass()) {
			return false;
		}
		SimpleMatrix matrix = (SimpleMatrix) o;
		return delegate.equals(matrix.delegate);
	}

}
