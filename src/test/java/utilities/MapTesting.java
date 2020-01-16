package utilities;

import java.util.Arrays;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.DoubleStream;
import org.junit.Test;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class MapTesting {

	double[][] d = new double[][]
		{
			{1.1}, {2.9}, {0.3},
			{0.000001}, {0.2323}, {0.51},
			{6}, {1}, {3}
		};

	private double cSum;

	private double w = 1000000;
	private double tot1;
	private double tot2;

	private final int size = 100;

	@Test
	public void testSpeed() {
		DenseMatrix k = Matrix.Factory.importFromArray(d);
		k = Matrix.Factory.randn(300, 300);

		for (double[] d : k.toDoubleArray()) {
			for (double l : d) {
				cSum += l;
			}
		}
		System.out.println(cSum);
		mapSingle(k, (Double e) -> e * w);
		mapMultiple(k, (Double e) -> e * w);

		System.out.println("Loop: " + tot2);
		System.out.println("Stream: " + tot1);
	}

	private void mapMultiple(final DenseMatrix k, Function<Double, Double> l) {
		double[][] lk = k.toDoubleArray();
		long t1, t2;
		double sum = 0;
		Supplier<DoubleStream> of = () -> Arrays.stream(lk).parallel()
			.flatMapToDouble(Arrays::stream);
		t1 = System.nanoTime();

		for (int t = 0; t < size; t++) {
			sum = of.get().map(l::apply).sum();
		}
		t2 = System.nanoTime();
		long nt = (t2 - t1) / size;
		tot1 = nt * 10e-9;
	}

	private void mapSingle(final DenseMatrix k, Function<Double, Double> l) {
		double[][] lk = k.toDoubleArray();
		double[][] out = new double[lk.length][lk[0].length];
		double sum = 0;
		long t1, t2;
		t1 = System.nanoTime();
		for (int i = 0; i < size; i++) {
			sum = 0;
			for (int t = 0; t < lk.length; t++) {
				for (int j = 0; j < lk[0].length; j++) {
					sum += l.apply(lk[t][j]);
				}
			}
		}
		t2 = System.nanoTime();

		long nt = (t2 - t1) / size;
		tot2 = nt * 10e-9;
	}

}
