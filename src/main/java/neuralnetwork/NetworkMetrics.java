package neuralnetwork;

import org.knowm.xchart.*;
import org.knowm.xchart.BitmapEncoder.BitmapFormat;
import org.knowm.xchart.style.Styler.ChartTheme;
import org.knowm.xchart.style.Styler.LegendPosition;

import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

public class NetworkMetrics {

    private static final String EPOCH = "Epoch";
    private final List<Integer> epochs;
    private final List<Double> lossValues;
    private final List<Double> correctValues;
    private final List<Double> calculationTimes;
    private final String matrixType;

    public NetworkMetrics(int epochs, String matrixType) {
        this.matrixType = matrixType;
        this.epochs = new ArrayList<>(epochs);
        this.lossValues = new ArrayList<>();
        this.correctValues = new ArrayList<>();
        this.calculationTimes = new ArrayList<>();
    }

    public NetworkMetrics(String matrixType) {
        this.matrixType = matrixType;
        this.epochs = new ArrayList<>();
        this.calculationTimes = new ArrayList<>();
        this.lossValues = new ArrayList<>();
        this.correctValues = new ArrayList<>();
    }

    /**
     * Adds plot data from the neural network.
     *
     * @param i       which epoch?
     * @param correct accuracy on validation
     * @param loss    loss on validation
     * @param time    time taken for validation
     */
    protected void addPlotData(int i, double correct, double loss, long time) {
        epochs.add(i);
        correctValues.add(correct);
        lossValues.add(loss);
        calculationTimes.add(time * 1e-9);

    }

    /**
     * Adds plot data from the neural network.
     *
     * @param correct accuracy on validation
     * @param loss    loss on validation
     */
    protected void initialPlotData(final double correct, final double loss) {
        epochs.add(0);
        correctValues.add(correct);
        lossValues.add(loss);
    }

    public void present(final String path) throws IOException {
        chartForLoss(path);

        calculationTimes.remove(0);
        calculationTimes.remove(1);
        calculationTimes.remove(2);

        chartForTimes(path);
        chartForAccuracy(path);
        chartForTimeMetrics(path);
    }

    private double maxTime() {
        return calculationTimes.stream().max(Double::compare)
                .orElseThrow(() -> new RuntimeException("Could not find a maximum of this list."));
    }

    private double minTime() {
        return calculationTimes.stream().min(Double::compare)
                .orElseThrow(() -> new RuntimeException("Could not find a minimum of this list."));
    }

    private double avgTime() {
        double sum = 0;
        for (double k : calculationTimes) {
            sum += k;
        }
        return sum / calculationTimes.size();
    }

    private double getStandardDeviation() {
        double meanOfDiffs = getVariance();
        return Math.sqrt(meanOfDiffs);
    }

    private double getVariance() {
        double mean = avgTime();
        double temp = 0;

        for (double val : calculationTimes) {
            // Step 2:
            double squrDiffToMean = Math.pow(val - mean, 2);

            // Step 3:
            temp += squrDiffToMean;
        }

        return temp / (double) calculationTimes.size();
    }

    private void chartForLoss(String out) throws IOException {

        double min = Collections.min(lossValues);
        double max = Collections.max(lossValues);
        XYChart lossChart = generateChart("Loss per Epoch", "Loss", "loss(x)", epochs, lossValues,
                Collections.max(epochs), min, max);

        String chartPath = createChartPathFromBasePath(out, "LossToEpochPlot");

        BitmapEncoder.saveBitmapWithDPI(lossChart, chartPath, BitmapFormat.PNG, 300);
    }

    private void chartForAccuracy(String out) throws IOException {
        XYChart accuracyChart = generateChart("Accuracy per Epoch", "Accuracy", "acc(x)", epochs, correctValues,
                Collections.max(epochs), 0, 1);

        BitmapEncoder.saveBitmapWithDPI(accuracyChart, createChartPathFromBasePath(out, "AccuracyToEpochPlot"),
                BitmapFormat.PNG, 300);

    }

    private void chartForTimes(String out) throws IOException {
        XYChart timeChart = generateChart("Time measure per Epoch", "Time", "time(x)", epochs.subList(4, epochs.size()),
                calculationTimes, Collections.max(epochs), minTime(), maxTime());

        BitmapEncoder.saveBitmapWithDPI(timeChart, createChartPathFromBasePath(out, "TimeMeasureToEpochPlot"),
                BitmapFormat.PNG, 300);

    }

    private void chartForTimeMetrics(final String path) throws IOException {
        CategoryChart cg = new CategoryChartBuilder().width(800).height(500).theme(ChartTheme.GGPlot2)
                .title("Time Metrics, for each epoch (s)").xAxisTitle("Statistical measures").yAxisTitle("Time Taken")
                .build();

        cg.getStyler().setLegendPosition(LegendPosition.InsideNW);
        cg.getStyler().setHasAnnotations(true);

        cg.addSeries("Measures", Arrays.asList("Average", "Max", "Min", "Stdev", "Var"),
                Arrays.asList(avgTime(), maxTime(), minTime(), getStandardDeviation(), getVariance()));

        BitmapEncoder.saveBitmapWithDPI(cg, createChartPathFromBasePath(path, "TimeMetricsPlot") + "_" + getNow(),
                BitmapFormat.PNG, 300);

    }

    private String createChartPathFromBasePath(String in, String out) {
        String use = in.endsWith("/") ? in : in + "/";

        return use + out + "_" + matrixType + "_" + getNow();
    }

    private String getNow() {
        String formattedDate;
        final DateFormat sdf = new SimpleDateFormat("dd-MM-yyyy", Locale.ENGLISH);
        formattedDate = sdf.format(new Date());
        formattedDate = formattedDate.replace("/", "-");
        return formattedDate;
    }

    private XYChart generateChart(final String heading, final String yLabel, final String function,
            final List<Integer> xValues, final List<Double> yValues, double maxX, double minY, double maxY) {

        final XYChart chart = new XYChartBuilder().width(600).height(400).title(heading)
                .xAxisTitle(NetworkMetrics.EPOCH).yAxisTitle(yLabel).build();
        chart.getStyler().setXAxisMin((double) 0);
        chart.getStyler().setXAxisMax(maxX);
        chart.getStyler().setYAxisMin(minY);
        chart.getStyler().setYAxisMax(maxY);
        chart.addSeries("label", xValues, yValues);
        return chart;
    }

}
