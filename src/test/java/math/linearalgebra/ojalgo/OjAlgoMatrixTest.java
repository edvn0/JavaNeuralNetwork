package math.linearalgebra.ojalgo;

import org.apache.log4j.BasicConfigurator;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class OjAlgoMatrixTest {

    @Before
    public void setUp() throws Exception {
        BasicConfigurator.configure();
    }

    @Test
    public void rows() {
        OjAlgoMatrix matrix = new OjAlgoMatrix(new double[][] { { 1, 1, 2 }, { 1, 3, 1 } });
        assertEquals(2, matrix.rows());
    }

    @Test
    public void cols() {
        OjAlgoMatrix matrix = new OjAlgoMatrix(new double[][] { { 1, 1, 2 }, { 1, 3, 1 } });
        assertEquals(3, matrix.cols());
    }

    @Test
    public void multiply() {
        OjAlgoMatrix id = new OjAlgoMatrix(new double[][] { { 1, 0 }, { 0, 1 } });
        OjAlgoMatrix out = new OjAlgoMatrix(new double[][] { { 2, 3 }, { 1, 5 } });
        OjAlgoMatrix expectedIdOut = new OjAlgoMatrix(out);

        OjAlgoMatrix matrix = new OjAlgoMatrix(new double[][] { { 1, 1 }, { 0, 1 } });
        OjAlgoMatrix otherMatrix = new OjAlgoMatrix(new double[][] { { 2, 3 }, { 1, 5 } });
        OjAlgoMatrix expectedMult = new OjAlgoMatrix(new double[][] { { 3, 8 }, { 1, 5 } });

        assertEquals(expectedIdOut, out.multiply(id));
        assertEquals(expectedMult, matrix.multiply(otherMatrix));
    }

    @Test
    public void testMultiply() {

        OjAlgoMatrix m1 = new OjAlgoMatrix(new double[] { 1, 5, -3, 5 });

        OjAlgoMatrix out1Matrix = m1.multiply(-2);
        OjAlgoMatrix out2Matrix = m1.multiply(0.01);

        assertEquals(new OjAlgoMatrix(new double[] { -2, -10, 6, -10 }), out1Matrix);
        assertEquals(new OjAlgoMatrix(new double[] { 0.01 * 1, 0.01 * 5, 0.01 * -3, 0.01 * 5 }), out2Matrix);
    }

    @Test
    public void add() {
    }

    @Test
    public void testAdd() {
    }

    @Test
    public void subtract() {
    }

    @Test
    public void divide() {
    }

    @Test
    public void map() {
    }

    @Test
    public void mapElements() {
        OjAlgoMatrix m = new OjAlgoMatrix(new double[][] { { 9, 1_000_000, 4 }, { 1, 16, 49 }, { 25, 81, 100 } });
        assertEquals(new OjAlgoMatrix(new double[][] { { 3, 1000, 2 }, { 1, 4, 7 }, { 5, 9, 10 } }),
                m.mapElements(Math::sqrt));
    }

    @Test
    public void mutableMapTest() {
        OjAlgoMatrix m = new OjAlgoMatrix(new double[][] { { 9, 1_000_000, 4 }, { 1, 16, 49 }, { 25, 81, 100 } });
        m.mapElementsMutable(Math::sqrt);

        assertEquals(new OjAlgoMatrix(new double[][] { { 3, 1000, 2 }, { 1, 4, 7 }, { 5, 9, 10 } }), m);
    }
}