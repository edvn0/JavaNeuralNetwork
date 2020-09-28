package math.activations;

import math.linearalgebra.ojalgo.OjAlgoMatrix;

public class SoftmaxFunction extends ActivationFunction {

    /**
     *
     */
    private static final long serialVersionUID = -5298468440584699205L;

    /**
     * Takes as input a vector of size NX1 and returns a SoftMax Vector of that
     * input.
     *
     * @param input input vector.
     * @return softmax vector.
     */
    private OjAlgoMatrix softMax(OjAlgoMatrix input) {
        if (input.cols() != 1) {
            throw new IllegalArgumentException("You can only perform SoftMax on a vector.");
        }

        OjAlgoMatrix max = this.max(input);
        OjAlgoMatrix z = input.subtract(max);
        double sum = z.mapElements(Math::exp).sum();

        return z.mapElements((e) -> Math.exp(e) / sum);
    }

    private OjAlgoMatrix max(OjAlgoMatrix input) {
        double max = input.max();
        return OjAlgoMatrix.zeroes(input.cols(), 1).add(max);
    }

    @Override
    public OjAlgoMatrix derivativeOnInput(final OjAlgoMatrix input, final OjAlgoMatrix out) {
        double xOut = input.multiply(out).sum();
        OjAlgoMatrix derive = out.subtract(xOut);
        return input.multiply(derive);
    }

    @Override
    public String getName() {
        return "SOFTMAX";
    }

    @Override
    public OjAlgoMatrix function(OjAlgoMatrix m) {
        return this.softMax(m);
    }

    @Override
    public OjAlgoMatrix derivative(OjAlgoMatrix m) {
        return null;
    }
}
