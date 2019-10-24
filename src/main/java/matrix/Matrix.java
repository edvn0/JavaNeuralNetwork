package matrix;

import java.util.function.UnaryOperator;

final public class Matrix {

	private final int M;             // number of rows
	private final int N;             // number of columns
	private final double[][] data;   // M-by-N array


	/**
	 * Create Matrix, M rows by N columns.
	 *
	 * @param M rows
	 * @param N columns
	 */
	public Matrix(int M, int N) {
		this.M = M;
		this.N = N;
		data = new double[M][N];
	}

	/**
	 * Create Matrix from double[][] arr.
	 *
	 * @param data double[][] array to create matrix from.
	 */
	public Matrix(double[][] data) {
		M = data.length;
		N = data[0].length;
		this.data = new double[M][N];
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				this.data[i][j] = data[i][j];
			}
		}
	}

	/**
	 * create and return a random M-by-N matrix with values between -1 and 1
	 *
	 * @param M rows
	 * @param N columns
	 * @return new Matrix with i,j \in (0,1)
	 */
	public static Matrix random(int M, int N) {
		Matrix A = new Matrix(M, N);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				A.data[i][j] = Math.random();
			}
		}
		return A;
	}

	/**
	 * Returns the identity NxN matrix.
	 *
	 * @param N size of the matrix.
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
	 *
	 */
	public static Matrix fromArray(double[] input) {
		Matrix m = new Matrix(input.length, 1);
		for (int i = 0; i < input.length; i++) {
			m.data[i][0] = input[i];
		}
		return m;
	}

	// swap rows i and j
	private void swap(int i, int j) {
		double[] temp = data[i];
		data[i] = data[j];
		data[j] = temp;
	}

	// create and return the transpose of the invoking matrix
	public Matrix transpose() {
		Matrix A = new Matrix(N, M);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				A.data[j][i] = this.data[i][j];
			}
		}
		return A;
	}

	// return C = A + B
	public Matrix add(Matrix B) {
		Matrix A = this;
		if (B.M != A.M || B.N != A.N) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		Matrix C = new Matrix(M, N);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				C.data[i][j] = A.data[i][j] + B.data[i][j];
			}
		}
		return C;
	}


	// return C = A - B
	public Matrix subtract(Matrix B) {
		Matrix A = this;
		if (B.M != A.M || B.N != A.N) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		Matrix C = new Matrix(M, N);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				C.data[i][j] = A.data[i][j] - B.data[i][j];
			}
		}
		return C;
	}

	// does A = B exactly?
	public boolean equals(Matrix B) {
		Matrix A = this;
		if (B.M != A.M || B.N != A.N) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				if (A.data[i][j] != B.data[i][j]) {
					return false;
				}
			}
		}
		return true;
	}

	// return C = A * B
	public Matrix multiply(Matrix B) {
		Matrix A = this;
		if (A.N != B.M) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		Matrix C = new Matrix(A.M, B.N);
		for (int i = 0; i < C.M; i++) {
			for (int j = 0; j < C.N; j++) {
				for (int k = 0; k < A.N; k++) {
					C.data[i][j] += (A.data[i][k] * B.data[k][j]);
				}
			}
		}
		return C;
	}

	public Matrix hadamard(Matrix B) {
		Matrix A = this;
		Matrix C = new Matrix(A.M, B.N);
		for (int i = 0; i < C.M; i++) {
			for (int j = 0; j < C.N; j++) {
				C.data[i][j] = A.data[i][j] * B.data[i][j];
			}
		}
		return C;
	}

	// print matrix to standard output
	public void show() {
		System.out.println("Matrix dimensions: " + M + " rows by " + N + " columns.");
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				System.out.printf("%9.4f ", data[i][j]);
			}
			System.out.println();
		}
	}

	public Matrix map(UnaryOperator<Double> function) {
		Matrix a = this;
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				double val = this.data[i][j];
				a.data[i][j] = function.apply(val);
			}
		}
		return a;
	}


	public double[] toArray() {
		double[] values = new double[M * N];
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				values[i] = this.data[i][j];
			}
		}
		return values;
	}

	public double[][] getData() {
		return this.data;
	}
}