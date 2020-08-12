package neuralnetwork;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.CategoryChart;
import org.knowm.xchart.CategoryChartBuilder;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.style.Styler.LegendPosition;

public class NetworkMetrics {

	private final ArrayList<Double> xValues = new ArrayList<>();
	private final ArrayList<Double> lossValues = new ArrayList<>();
	private final ArrayList<Double> correctValues = new ArrayList<>();
	private final ArrayList<Double> calculationTimes = new ArrayList<>();

	public NetworkMetrics() {
	}

	/**
	 * Adds plot data from the neural network.
	 *
	 * @param i       which epoch?
	 * @param correct accuracy on validation
	 * @param loss    loss on validation
	 * @param time    time taken for validation
	 */
	protected void addPlotData(int i, Double correct, Double loss, Double time) {
		xValues.add((double) i);
		correctValues.add(correct);
		lossValues.add(loss);
		calculationTimes.add(time * 10E-6);
	}

	/**
	 * Adds plot data from the neural network.
	 *
	 * @param i       which epoch
	 * @param correct accuracy on validation
	 * @param loss    loss on validation
	 */
	protected void addPlotData(final int i, final double correct, final double loss) {
		xValues.add((double) i);
		correctValues.add(correct);
		lossValues.add(loss);
	}

	public void present(final String path) throws IOException {
		chartForLoss(path);
		chartForTimes(path);
		chartForAccuracy(path);
		chartForTimeMetrics(path);
	}

	private double maxTime() {
		return Collections.max(calculationTimes.stream().map(e -> 1e-9).collect(Collectors.toList()));
	}

	private double minTime() {
		return Collections.min(calculationTimes.stream().map(e -> 1e-9).collect(Collectors.toList()));
	}

	private double avgTime() {
		double sum = 0;
		for (Double k : calculationTimes) {
			sum += k;
		}
		return sum / calculationTimes.size();
	}

	public void appendEpochValues(Double k) {
		xValues.add(k);
	}

	public void appendLossValues(Double k) {
		lossValues.add(k);
	}

	public void appendCorrectValues(Double k) {
		correctValues.add(k);
	}

	public void appendCalculationTimes(Double k) {
		calculationTimes.add(k);
	}

	private void chartForLoss(String out) throws IOException {
		BitmapEncoder.saveBitmapWithDPI(
				generateChart("Loss per Epoch", "Epoch", "Loss", "loss(x)", xValues, lossValues, 0,
						Collections.max(xValues), 0, Collections.max(lossValues)),
				createChartPathFromBasePath(out, "LossToEpochPlot") + "_" + getNow(), BitmapFormat.PNG, 300);
	}

	private void chartForAccuracy(String out) throws IOException {
		BitmapEncoder.saveBitmapWithDPI(
				generateChart("Accuracy per Epoch", "Epoch", "Accuracy", "acc(x)", xValues, correctValues, 0,
						Collections.max(xValues), 0, 1),
				createChartPathFromBasePath(out, "AccuracyToEpochPlot") + "_" + getNow(), BitmapFormat.PNG, 300);

	}

	private void chartForTimes(String out) throws IOException {
		BitmapEncoder.saveBitmapWithDPI(
				generateChart("Time measure per Epoch", "Epoch", "Time", "time(x)", xValues.subList(1, xValues.size()),
						calculationTimes.stream().map(e -> 1e-6 * e).collect(Collectors.toList()), 0,
						Collections.max(xValues), minTime(), maxTime()),
				createChartPathFromBasePath(out, "TimeMeasureToEpochPlot") + "_" + getNow(), BitmapFormat.PNG, 300);

	}

	private void chartForTimeMetrics(final String path) throws IOException {
		CategoryChart cg = new CategoryChartBuilder().width(800).height(500).title("Time Metrics")
				.xAxisTitle("Time Categories").yAxisTitle("Time Taken").build();

		cg.getStyler().setLegendPosition(LegendPosition.InsideNW);
		cg.getStyler().setHasAnnotations(true);

		cg.addSeries("Measures", Arrays.asList("Average", "Max", "Min"),
				Arrays.asList(avgTime(), maxTime(), minTime()));

		BitmapEncoder.saveBitmapWithDPI(cg, createChartPathFromBasePath(path, "TimeMetricsPlot") + "_" + getNow(),
				BitmapFormat.PNG, 300);

	}

	private String createChartPathFromBasePath(String in, String out) {
		final String use = in.endsWith("/") ? in : in + "/";
		return use + out;
	}

	private String getNow() {
		String formattedDate;
		final SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss", Locale.ENGLISH);
		formattedDate = sdf.format(new Date());

		return formattedDate;
	}

	private XYChart generateChart(final String heading, final String xLabel, final String yLabel, final String function,
			final List<Double> xValues, final List<Double> yValues, double minX, double maxX, double minY,
			double maxY) {
		final XYChart chart = QuickChart.getChart(heading, xLabel, yLabel, function, xValues, yValues);
		chart.getStyler().setXAxisMin(minX);
		chart.getStyler().setXAxisMax(maxX);
		chart.getStyler().setYAxisMin(minY);
		chart.getStyler().setYAxisMax(maxY);
		return chart;
	}

}
