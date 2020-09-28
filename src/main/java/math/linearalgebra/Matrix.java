package math.linearalgebra;

import java.util.function.Function;

public interface Matrix<M> {

    int rows();

    int cols();

    /**
     * Matrix multiplication, should throw if cols and rows do not match.
     * Contract is This X in, i.e. this_rows*this_cols X in_cols*in_rows
     *
     * @param otherMatrix right operand
     * @return new matrix multiplied
     */
    M multiply(M otherMatrix);

    /**
     * Multiply each element with this scalar
     *
     * @param scalar to multiply with
     * @return scaled with scalar
     */
    M multiply(double scalar);

    M add(M in);

    M add(double in);

    M subtract(double in);

    M subtract(M in);

    M divide(double in);

    double map(Function<M, Double> mapping);

    M mapElements(Function<Double, Double> mapping);

    double sum();

    double max();

    M transpose();
}
