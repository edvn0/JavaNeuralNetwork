package math.linearalgebra;

import math.linearalgebra.ojalgo.OjAlgoMatrix;
import math.linearalgebra.ujmp.UJMPMatrix;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class GenericsTest {


    private static Matrix<?> matrix;
    private static Matrix<?> matrix2;

    @Before
    public void before() {
        matrix = new OjAlgoMatrix(new double[]{1, 2, 3}, 3, 1);
        matrix2 = new UJMPMatrix(new double[]{1, 2, 3}, 3, 1);
    }

    @Test
    public void testGenerics() {
        var out = matrix.add(5);
        var out2 = matrix2.add(5);

        Matrix<?> correctUJMP = new UJMPMatrix(new double[]{6, 7, 8}, 3, 1);
        Matrix<?> correctOjAlgo = new OjAlgoMatrix(new double[]{6, 7, 8}, 3, 1);

        assertEquals(correctOjAlgo, out);
        assertEquals(correctUJMP, out2);
    }

    @Test
    public void testNN() {
        Matrix<? extends Matrix<?>> input = new OjAlgoMatrix(new double[]{1, 1}, 2, 1);// Vector, 2X1
    }

}
