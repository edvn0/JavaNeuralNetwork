package math.linearalgebra.ujmp;

import lombok.extern.slf4j.Slf4j;
import math.linearalgebra.Matrix;
import org.apache.log4j.BasicConfigurator;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@Slf4j
public class UJMPMatrixTest {
    @Before
    public void setUp() {
        BasicConfigurator.configure();
    }

    @Test
    public void create() {
        Matrix<UJMPMatrix> twoByThree = new UJMPMatrix(new double[]{1, 1, 2, 1, 3, 1}, 2, 3);

        assertEquals(2, twoByThree.rows());
        assertEquals(3, twoByThree.cols());

        Matrix<UJMPMatrix> threeByTwo = new UJMPMatrix(new double[]{1, 1, 2, 1, 3, 1}, 3, 2);

        assertEquals(3, threeByTwo.rows());
        assertEquals(2, threeByTwo.cols());
    }

    @Test
    public void multiply() {

        Matrix<UJMPMatrix> id = new UJMPMatrix(null,math.linearalgebra.Matrix.MatrixType.IDENTITY, 2,2);
        Matrix<UJMPMatrix> out = new UJMPMatrix(new double[]{2, 3, 1, 5}, 2, 2);
        Matrix<UJMPMatrix> expectedIdOut = new UJMPMatrix(out);

        Matrix<UJMPMatrix> matrix = new UJMPMatrix(new double[]{1, 1, 0, 1}, 2, 2);
        Matrix<UJMPMatrix> otherMatrix = new UJMPMatrix(new double[]{2, 3, 1, 5}, 2, 2);
        Matrix<UJMPMatrix> expectedMult = new UJMPMatrix(new double[]{2, 5, 1, 6}, 2, 2);

        assertEquals(expectedIdOut, out.multiply(id));
        assertEquals(expectedMult, matrix.multiply(otherMatrix));

    }

    @Test
    public void testMultiply() {

        Matrix<UJMPMatrix> ones = new UJMPMatrix(null, Matrix.MatrixType.ONES, 2,2);
        Matrix<UJMPMatrix> fives = ones.multiply(5d);

        Matrix<UJMPMatrix> fiveExpected = new UJMPMatrix(new double[][]{{5d,5d},{5d,5d}}, 2,2);

        assertEquals(ones.cols(), fives.cols());
        assertEquals(ones.rows(), fives.rows());
        assertEquals(fiveExpected, fives);
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

        Matrix<UJMPMatrix> valsNine = new UJMPMatrix(new double[]{1,2,-4,10}, 2,2);
        Matrix<UJMPMatrix> valsMinus6 = new UJMPMatrix(new double[]{0,0,-6,0,0,0}, 2,3);
        Matrix<UJMPMatrix> vals100 = new UJMPMatrix(new double[]{10,10,10,10,10,10,10,10,20}, 3,3);



        double outNine = valsNine.map(Matrix::sum);
        double outMinus6 = valsMinus6.map(Matrix::sum);
        double out100 = vals100.map(Matrix::sum);

        assertEquals(9, outNine, 1e-5);
        assertEquals(-6, outMinus6, 1e-5);
        assertEquals(100, out100, 1e-5);

    }

    @Test
    public void mapElements() {

        Matrix<UJMPMatrix> matrix = new UJMPMatrix(new double[]{2,2,1,10},2,2);

        assertEquals(matrix.map(Matrix::sum), 15, 1e-5);
        Matrix<UJMPMatrix> mapped = matrix.mapElements(e -> e * 10);

        assertEquals(new UJMPMatrix(new double[]{20,20,10,100},2,2), mapped);
        assertEquals(mapped.map(Matrix::sum), 150, 1e-5);

    }
}