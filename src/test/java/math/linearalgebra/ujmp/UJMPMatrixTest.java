package math.linearalgebra.ujmp;

import static org.junit.Assert.assertEquals;

import org.apache.log4j.BasicConfigurator;
import org.junit.Before;
import org.junit.Test;

public class UJMPMatrixTest {

    @Before
    public void setUp() throws Exception {
        BasicConfigurator.configure();
    }

    @Test
    public void rows() {
        UJMPMatrix matrix = new UJMPMatrix(new double[][] { { 1, 1, 2 }, { 1, 3, 1 } });
        assertEquals(2, matrix.rows());
    }

    @Test
    public void cols() {
        UJMPMatrix matrix = new UJMPMatrix(new double[][] { { 1, 1, 2 }, { 1, 3, 1 } });
        assertEquals(3, matrix.cols());
    }

    @Test
    public void multiply() {
        UJMPMatrix id = new UJMPMatrix(new double[][] { { 1, 0 }, { 0, 1 } });
        UJMPMatrix out = new UJMPMatrix(new double[][] { { 2, 3 }, { 1, 5 } });
        UJMPMatrix expectedIdOut = new UJMPMatrix(out);

        UJMPMatrix matrix = new UJMPMatrix(new double[][] { { 1, 1 }, { 0, 1 } });
        UJMPMatrix otherMatrix = new UJMPMatrix(new double[][] { { 2, 3 }, { 1, 5 } });
        UJMPMatrix expectedMult = new UJMPMatrix(new double[][] { { 3, 8 }, { 1, 5 } });

        assertEquals(expectedIdOut, out.multiply(id));
        assertEquals(expectedMult, matrix.multiply(otherMatrix));
    }

    @Test
    public void testMultiply() {

        UJMPMatrix m1 = new UJMPMatrix(new double[][] { { 1, 5 }, { -3, 5 } });

        UJMPMatrix out1Matrix = m1.multiply(-2);
        UJMPMatrix out2Matrix = m1.multiply(0.01);

        assertEquals(new UJMPMatrix(new double[][] { { -2, -10 }, { 6, -10 } }), out1Matrix);
        assertEquals(new UJMPMatrix(new double[][] { { 0.01 * 1, 0.01 * 5 }, { 0.01 * -3, 0.01 * 5 } }), out2Matrix);
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
    }

    @Test
    public void mutableMapTest() {
        UJMPMatrix m = new UJMPMatrix(new double[][] { { 9, 1_000_000, 4 }, { 1, 16, 49 }, { 25, 81, 100 } });
        m.mapElementsMutable(Math::sqrt);

        assertEquals(new UJMPMatrix(new double[][] { { 3, 1000, 2 }, { 1, 4, 7 }, { 5, 9, 10 } }), m);
    }
}