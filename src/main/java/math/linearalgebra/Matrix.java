package math.linearalgebra;

import java.util.function.Function;

public interface Matrix<M> {

    default Matrix<M> squareMatrix() {
        return this.multiply(this);
    }

    /**
     * Cols or this matrix
     *
     * @return columns
     */
    public int rows();

    /**
     * Rows of this matrix
     *
     * @return rows
     */
    public int cols();

    /**
     * Matrix multiplication, should throw if cols and rows do not match. Contract
     * is This X in, i.e. this_rows*this_cols X in_cols*in_rows
     *
     * @param otherMatrix right operand
     * @return new matrix multiplied
     */
    public Matrix<M> multiply(Matrix<M> otherMatrix);

    /**
     * Element wise multiplication of two matrices.
     * 
     * @param otherMatrix right operand
     * @return new element wise multiplied matrix
     */
    public Matrix<M> hadamard(Matrix<M> otherMatrix);

    /**
     * Multiply each element with this scalar
     *
     * @param scalar to multiply with
     * @return scaled with scalar
     */
    public Matrix<M> multiply(double scalar);

    /**
     * Add in to this matrix
     *
     * @param in right operand
     * @return this + in
     */
    public Matrix<M> add(Matrix<M> in);

    /**
     * Add in to all elements of this.
     *
     * @param in scalar operand
     * @return this.map(e - > e + in)
     */
    public Matrix<M> add(double in);

    /**
     * Subtract in from all elements of this
     *
     * @param in scalar operand
     * @return this.map(e - > e - in);
     */
    public Matrix<M> subtract(double in);

    /**
     * Substract in from this matrix
     *
     * @param in right operand
     * @return this[i][j] -= in[i][j]
     */
    public Matrix<M> subtract(Matrix<M> in);

    /**
     * Divide all elements by in
     *
     * @param in scalar operand
     * @return in.map(e - > e / in);
     */
    public Matrix<M> divide(double in);

    /**
     * Map this matrix to a double, useful for reduce or trace implementations
     *
     * @param mapping f: This -> double
     * @return a double value
     */
    public double map(Function<Matrix<M>, Double> mapping);

    /**
     * Map each element with this function
     *
     * @param mapping f: Double -> Double each element
     * @return this.map(e - > mapping ( e));
     */
    public Matrix<M> mapElements(Function<Double, Double> mapping);

    /**
     * Sum this matrix over all entries.
     *
     * @return sum of this
     */
    public double sum();

    /**
     * Max of this matrix over all entries.
     *
     * @return max of this
     */
    public double max();

    /**
     * Index along a column of max, should only be used for vectors.
     *
     * @return index of max
     */
    public int argMax();

    /**
     * Transpose this matrix.
     *
     * @return transpose.
     */
    public Matrix<M> transpose();

    public M delegate();

    public Matrix<M> divide(Matrix<M> right);

    public Matrix<M> maxVector();

    public Matrix<M> zeroes(int rows, int cols);

    public Matrix<M> ones(int rows, int cols);

    public Matrix<M> identity(int rows, int cols);

    public String toString();

    public enum MatrixType {
        VECTOR, SQUARE, ZEROES, ONES, IDENTITY
    }

    public double square();

    public String name();
}
