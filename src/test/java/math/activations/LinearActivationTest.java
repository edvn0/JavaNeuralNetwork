package math.activations;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import math.linearalgebra.Matrix;
import math.linearalgebra.simple.SMatrix;
import math.linearalgebra.simple.SimpleMatrix;

public class LinearActivationTest {

    @Test
    public void testLinear() {

        ActivationFunction<SMatrix> s = new LinearFunction<>(1);

        double[][] vars = { { 1 }, { 2 }, { 3 } };
        SimpleMatrix m = new SimpleMatrix(vars);

        Matrix<SMatrix> exc = s.function(m);

        assertEquals("Linear function should just do nothing", m.delegate(), exc.delegate());

    }

}
