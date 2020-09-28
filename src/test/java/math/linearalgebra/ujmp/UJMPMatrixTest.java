package math.linearalgebra.ujmp;

import lombok.extern.slf4j.Slf4j;
import math.linearalgebra.ujmp.UJMPMatrix;
import org.apache.log4j.BasicConfigurator;
import org.junit.Before;
import org.junit.Test;
import org.ujmp.core.Matrix;

import java.util.Arrays;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@Slf4j
public class UJMPMatrixTest {
    static Function<UJMPMatrix, Double> mapper = (e) -> Arrays.stream(e.getDelegate().toDoubleArray()).flatMapToDouble(b -> Arrays.stream(b.clone())).sum();

    @Before
    public void setUp() {
        BasicConfigurator.configure();
    }

    @Test
    public void create() {
        UJMPMatrix twoByThree = new UJMPMatrix(new double[]{1, 1, 2, 1, 3, 1}, 2, 3);

        assertEquals(2, twoByThree.rows());
        assertEquals(3, twoByThree.cols());

        UJMPMatrix threeByTwo = new UJMPMatrix(new double[]{1, 1, 2, 1, 3, 1}, 3, 2);

        assertEquals(3, threeByTwo.rows());
        assertEquals(2, threeByTwo.cols());
    }

    @Test
    public void multiply() {

        UJMPMatrix id = UJMPMatrix.identity(2,2);
        UJMPMatrix out = new UJMPMatrix(new double[]{2, 3, 1, 5}, 2, 2);
        UJMPMatrix expectedIdOut = new UJMPMatrix(out);

        UJMPMatrix matrix = new UJMPMatrix(new double[]{1, 1, 0, 1}, 2, 2);
        UJMPMatrix otherMatrix = new UJMPMatrix(new double[]{2, 3, 1, 5}, 2, 2);
        UJMPMatrix expectedMult = new UJMPMatrix(new double[]{2, 5, 1, 6}, 2, 2);

        assertEquals(expectedIdOut, out.multiply(id));
        assertEquals(expectedMult, matrix.multiply(otherMatrix));

    }

    @Test
    public void testMultiply() {

        UJMPMatrix ones = new UJMPMatrix(Matrix.Factory.ones(2, 2));
        UJMPMatrix fives = ones.multiply(5d);

        assertEquals(ones.cols(), fives.cols());
        assertEquals(ones.rows(), fives.rows());
        assertArrayEquals(fives.getDelegate().toDoubleArray(), new double[][]{{5d, 5d}, {5d, 5d}});
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

        UJMPMatrix valsNine = new UJMPMatrix(new double[]{1,2,-4,10}, 2,2);
        UJMPMatrix valsMinus6 = new UJMPMatrix(new double[]{0,0,-6,0,0,0}, 2,3);
        UJMPMatrix vals100 = new UJMPMatrix(new double[]{10,10,10,10,10,10,10,10,20}, 3,3);

        double outNine = mapper.apply(valsNine);
        double outMinus6 = mapper.apply(valsMinus6);
        double out100 = mapper.apply(vals100);

        assertEquals(9, outNine, 1e-5);
        assertEquals(-6, outMinus6, 1e-5);
        assertEquals(100, out100, 1e-5);

    }

    @Test
    public void mapElements() {

        UJMPMatrix matrix = new UJMPMatrix(new double[]{2,2,1,10},2,2);

        assertEquals(matrix.map(mapper), 15, 1e-5);
        UJMPMatrix mapped = matrix.mapElements(e -> {
            int rows = e.length;
            int cols = e[0].length;
            double[] out = new double[rows*cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    out[j*cols+i] = e[i][j]*10;
                }
            }
            return new UJMPMatrix(out, rows, cols);
        });

        assertEquals(new UJMPMatrix(new double[]{20,20,10,100},2,2), mapped);
        assertEquals(mapped.map(mapper), 150, 1e-5);

    }
}