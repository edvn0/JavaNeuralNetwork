package matrix;

import java.io.Serializable;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.UnaryOperator;

final public class Matrix implements Serializable {

	private final int rows;             // number of rows
	private final int columns;             // number of columns
	private final double[][] data;   // M-by-N array


	/**
	 * Create Matrix, M rows by N columns.
	 *
	 * @param rows rows
	 * @param N    columns
	 */
	public Matrix(int rows, int N) {
		this.rows = rows;
		this.columns = N;
		data = new double[this.rows][this.columns];
	}

	/**
	 * Create Matrix from double[][] arr.
	 *
	 * @param data double[][] array to create matrix from.
	 */
	public Matrix(double[][] data) {
		rows = data.length;
		columns = data[0].length;
		this.data = new double[rows][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				this.data[i][j] = data[i][j];
			}
		}
	}

	/**
	 * create and return a random M-by-N matrix with values between -1 and 1
	 *
	 * @param M rows
	 * @param N columns
	 *
	 * @return new Matrix with i,j \in (-1,1)
	 */
	public static Matrix random(int M, int N) {
		Matrix A = new Matrix(M, N);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				A.data[i][j] = 2 * Math.random() - 1;
			}
		}
		return A;
	}

	/**
	 * create and return a random M-by-N matrix with values between -1 and 1
	 *
	 * @param M rows
	 * @param N columns
	 *
	 * @return new Matrix with i,j \in (-1,1)
	 */
	public static Matrix randomFromRange(int M, int N, double a, double b) {
		Matrix A = new Matrix(M, N);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				A.data[i][j] = ThreadLocalRandom.current().nextDouble(a, b);
			}
		}
		return A;
	}


	/**
	 * Returns the identity NxN matrix.
	 *
	 * @param N size of the matrix.
	 *
	 * @return The identity NxN matrix.
	 */
	public static Matrix identity(int N) {
		Matrix I = new Matrix(N, N);
		for (int i = 0; i < N; i++) {
			I.data[i][i] = 1;
		}
		return I;
	}

	/**
	 * Creates a vector (Mx1 Matrix) from double array
	 *
	 * @param input double array
	 *
	 * @return Matrix of size (input length x 1)
	 */
	public static Matrix fromArray(double[] input) {
		Matrix m = new Matrix(input.length, 1);
		for (int i = 0; i < input.length; i++) {
			m.data[i][0] = input[i];
		}
		return m;
	}

	// create and return the transpose of the invoking matrix
	public Matrix transpose() {
		Matrix A = new Matrix(columns, rows);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				A.data[j][i] = this.data[i][j];
			}
		}
		return A;
	}

	public double matrixSum() {
		double[][] data = this.data;
		double k = 0;
		for (double[] datum : data) {
			for (double d : datum) {
				k += d;
			}
		}
		return k;
	}

	// return C = A + B
	public Matrix add(Matrix B) {
		Matrix A = this;
		if (B.rows != A.rows || B.columns != A.columns) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		Matrix C = new Matrix(rows, columns);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				C.data[i][j] = A.data[i][j] + B.data[i][j];
			}
		}
		return C;
	}

	// return C = A - B, C = A + (-B)
	public Matrix subtract(Matrix B) {
		Matrix A = this;
		if (B.rows != A.rows || B.columns != A.columns) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		Matrix C = new Matrix(rows, columns);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				C.data[i][j] = A.data[i][j] - B.data[i][j];
			}
		}
		return C;
	}

	/**
	 * Return C = this * B
	 *
	 * @param B other Matrix
	 *
	 * @return C = this * B;
	 */
	public Matrix multiply(Matrix B) {
		Matrix A = this;
		if (A.columns != B.rows) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		Matrix C = new Matrix(A.rows, B.columns);
		for (int i = 0; i < C.rows; i++) {
			for (int j = 0; j < C.columns; j++) {
				for (int k = 0; k < A.columns; k++) {
					C.data[i][j] += (A.data[i][k] * B.data[k][j]);
				}
			}
		}
		return C;
	}

	public double dotProduct(Matrix B) {
		if (this.getRows() != B.getRows() || !(this.getColumns() == 1 && B.getColumns() == 1)) {
			throw new IllegalArgumentException("Matrices are not vectors.");
		}

		double sum = 0;
		for (int i = 0; i < B.data.length; i++) {
			sum += this.data[i][0] * B.data[i][0];
		}

		return sum;
	}

	public Matrix hadamard(Matrix B) {
		Matrix A = this;
		Matrix C = new Matrix(A.rows, B.columns);
		for (int i = 0; i < C.rows; i++) {
			for (int j = 0; j < C.columns; j++) {
				C.data[i][j] = A.data[i][j] * B.data[i][j];
			}
		}
		return C;
	}

	public Matrix negate(Matrix A) {
		Matrix B = new Matrix(A.rows, A.columns);
		for (int i = 0; i < B.rows; i++) {
			for (int j = 0; j < B.columns; j++) {
				B.data[i][j] = -1 * A.data[i][j];
			}
		}
		return B;
	}

	/**
	 * String representation of the Matrix
	 *
	 * @return Matrix as a string
	 */
	public String stringRepresentation() {
		StringBuilder output = new StringBuilder();
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				output.append(String.format("%9.4f ", data[i][j]));
			}
			output.append("\n");
		}
		return output.toString();
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("Matrix{");
		sb.append("rows=").append(rows);
		sb.append(", columns=").append(columns);
		sb.append(", data=").append(Arrays.deepToString(data));
		sb.append('}');
		return sb.toString();
	}

	/**
	 * Applies a function {@link UnaryOperator} to each element
	 *
	 * @param function function to be applied
	 *
	 * @return Matrix with mapped values
	 */
	public Matrix map(UnaryOperator<Double> function) {
		Matrix a = this;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				double val = this.data[i][j];
				double after = function.apply(val);
				a.data[i][j] = after;
			}
		}
		return a;
	}

	/**
	 * Takes this matrix object and returns a flat array which "wraps" with the column size.
	 *
	 * @return flat array representation of Matrix
	 */
	public double[] toArray() {
		double[] values = new double[rows * columns];
		int k = 0;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				values[k++] = this.data[i][j];
			}
		}
		return values;
	}

	public double[][] getData() {
		return this.data;
	}

	public int getColumns() {
		return columns;
	}

	public int getRows() {
		return rows;
	}

	/**
	 * Helper method to return a value if it is a 1x1 Matrix.
	 *
	 * @return double representing a scalar. data[0][0]
	 */
	public double getSingleValue() {
		return this.data[0][0];
	}


	// Unfortunately mutates this object...
	public void setData(Matrix newWeights) {

		int rows = this.getRows();
		int cols = this.getColumns();

		if (newWeights.getRows() != this.getRows() || newWeights.getColumns() != this
			.getColumns()) {
			throw new InputMismatchException("Matrix dimensions do not match");
		}

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				this.data[i][j] = newWeights.data[i][j];
			}
		}
	}

	public double magnitude() {
		if (columns > 1) {
			throw new IllegalArgumentException(
				"This is not a vector, you cannot take its magnitude.");
		}
		double sum = 0;
		int its = 0;
		for (double[] datum : this.data) {
			sum += Math.pow(datum[0], 2);
			its++;
		}
		return Math.pow(sum, 1d / its);
	}

	// does A == B exactly?
	@Override
	public boolean equals(Object o) {
		Matrix A = this;
		if (this == o) {
			return true;
		}
		if (!(o instanceof Matrix)) {
			return false;
		}

		Matrix B = (Matrix) o;

		if (B.rows != A.rows || B.columns != A.columns) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				double a = A.getData()[i][j];
				double b = B.getData()[i][j];
				if (Double.compare(a, b) != 0) {
					return false;
				}
			}
		}
		return true;
	}

	public String getDimension() {
		return "[" + this.rows + " X " + this.columns + "]";
	}

	public double getElement(int i, int j) {
		return this.data[i][j];
	}
}