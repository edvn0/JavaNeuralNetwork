package math.linearalgebra;

import java.util.function.Function;
import java.util.function.UnaryOperator;

public interface NeuralNetworkMatrix<M, T extends Number> {

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
    M multiply(T scalar);

    M add(M in);

    M add(T in);

    M subtract(T in);

    M divide(T in);

    T map(Function<M, T> mapping);

    M mapElements(Function<double[][], M> mapping);


}
