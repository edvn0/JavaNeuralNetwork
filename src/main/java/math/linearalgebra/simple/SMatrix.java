package math.linearalgebra.simple;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

import utilities.exceptions.MatrixException;

public class SMatrix {
    private final int M; // number of rows
    private final int N; // number of columns
    private final double[][] data; // M-by-N array

    // create M-by-N matrix of 0's
    public SMatrix(int M, int N) {
        this.M = M;
        this.N = N;
        data = new double[M][N];
    }

    // create matrix based on 2d array
    public SMatrix(double[][] vals) {
        M = vals.length;
        N = vals[0].length;
        this.data = new double[M][N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                this.data[i][j] = vals[i][j];
    }

    public SMatrix(double[] ds) {
        this.M = ds.length;
        this.N = 1;
        this.data = new double[M][N];
        for (int i = 0; i < M; i++) {
            this.data[i][0] = ds[i];
        }
    }

    private SMatrix applyOperator(Function<Double, Double> in) {
        SMatrix A = this;
        SMatrix out = new SMatrix(A.M, A.N);
        for (int i = 0; i < out.M; i++) {
            for (int j = 0; j < out.N; j++) {
                out.data[i][j] = in.apply(A.data[i][j]);
            }
        }
        return out;
    }

    private SMatrix applyOperator(BiFunction<Double, Double, Double> in, SMatrix B) {
        SMatrix A = this;
        SMatrix out = new SMatrix(A.M, A.N);
        for (int i = 0; i < out.M; i++) {
            for (int j = 0; j < out.N; j++) {
                out.data[i][j] = in.apply(A.data[i][j], B.data[i][j]);
            }
        }
        return out;
    }

    // create and return a random M-by-N matrix with values between 0 and 1
    public static SMatrix random(int M, int N) {
        SMatrix A = new SMatrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A.data[i][j] = Math.random();
        return A;
    }

    // create and return the N-by-N identity matrix
    public static SMatrix identity(int N) {
        SMatrix I = new SMatrix(N, N);
        for (int i = 0; i < N; i++)
            I.data[i][i] = 1;
        return I;
    }

    // create and return the transpose of the invoking matrix
    public SMatrix transpose() {
        SMatrix A = new SMatrix(N, M);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A.data[j][i] = this.data[i][j];
        return A;
    }

    // return C = A + B
    public SMatrix plus(SMatrix B) {
        SMatrix A = this;
        if (B.M != A.M || B.N != A.N)
            throw new RuntimeException("Illegal matrix dimensions.");
        SMatrix C = new SMatrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C.data[i][j] = A.data[i][j] + B.data[i][j];
        return C;
    }

    // return C = A - B
    public SMatrix minus(SMatrix B) {
        SMatrix A = this;
        if (B.M != A.M || B.N != A.N)
            throw new RuntimeException("Illegal matrix dimensions.");
        SMatrix C = new SMatrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C.data[i][j] = A.data[i][j] - B.data[i][j];
        return C;
    }

    // does A = B exactly?
    @Override
    public boolean equals(Object o) {
        SMatrix B = (SMatrix) o;
        SMatrix A = this;
        if (B.M != A.M || B.N != A.N)
            throw new RuntimeException("Illegal matrix dimensions.");
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                if (A.data[i][j] != B.data[i][j])
                    return false;
        return true;
    }

    // return C = A * B
    public SMatrix times(SMatrix B) {
        SMatrix A = this;
        if (A.N != B.M)
            throw new RuntimeException("Illegal matrix dimensions.");
        SMatrix C = new SMatrix(A.M, B.N);
        for (int i = 0; i < C.M; i++)
            for (int j = 0; j < C.N; j++)
                for (int k = 0; k < A.N; k++)
                    C.data[i][j] += (A.data[i][k] * B.data[k][j]);
        return C;
    }

    public SMatrix times(double val) {
        SMatrix A = this;
        SMatrix C = new SMatrix(A.M, A.N);
        for (int i = 0; i < C.M; i++)
            for (int j = 0; j < C.N; j++)
                for (int k = 0; k < A.N; k++)
                    C.data[i][j] = (A.data[i][k] * val);
        return C;
    }

    public int rows() {
        return this.M;
    }

    public int cols() {
        return this.N;
    }

    public SMatrix hadamard(SMatrix B) {
        return applyOperator((a, b) -> a * b, B);
    }

    public SMatrix plus(double in) {
        return applyOperator(e -> e + in);
    }

    public SMatrix minus(double in) {
        return applyOperator(e -> e - in);
    }

    public SMatrix divide(double in) {
        return applyOperator(e -> e / in);
    }

    public double sum() {
        double sum = 0;
        for (int i = 0; i < this.M; i++) {
            for (int j = 0; j < this.N; j++) {
                sum += data[i][j];
            }
        }
        return sum;
    }

    public double max() {
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < this.M; i++) {
            for (int j = 0; j < this.N; j++) {
                if (data[i][j] > max) {
                    max = data[i][j];
                }
            }
        }
        return max;
    }

    public int argMax() {
        double max = -Double.MAX_VALUE;
        int argMax = -1;
        for (int i = 0; i < this.M; i++) {
            if (data[i][0] > max) {
                max = data[i][0];
                argMax = i;
            }
        }
        return argMax;
    }

    public SMatrix divide(SMatrix delegate) {
        SMatrix A = this;
        if (delegate.M != A.M || delegate.N != A.N)
            throw new RuntimeException("Illegal matrix dimensions.");
        SMatrix C = new SMatrix(M, N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C.data[i][j] = A.data[i][j] / delegate.data[i][j];
        return C;
    }

    public SMatrix maxVector() {
        double[][] values = new double[M][1];
        double max = this.max();
        for (int i = 0; i < M; i++) {
            values[i][0] = max;
        }
        return new SMatrix(values);
    }

    public double norm() {
        if (this.cols() != 1) {
            throw new MatrixException("Not a vector.");
        }

        SMatrix squared = applyOperator(e -> e * e);
        return Math.sqrt(squared.sum());
    }

    public double[][] rawCopy() {
        if (data == null)
            return null;
        double[][] result = new double[data.length][];
        for (int r = 0; r < data.length; r++) {
            result[r] = data[r].clone();
        }
        return result;
    }

    @Override
    public String toString() {
        return Arrays.deepToString(this.data);
    }
}
