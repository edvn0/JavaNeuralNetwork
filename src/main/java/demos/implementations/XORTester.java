package demos.implementations;

import math.activations.TanhFunction;
import math.error_functions.MeanSquaredCostFunction;
import math.evaluation.ThresholdEvaluationFunction;
import math.linearalgebra.ojalgo.OjAlgoMatrix;
import neuralnetwork.NetworkBuilder;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.initialiser.InitialisationMethod;
import neuralnetwork.initialiser.OjAlgoFactory;
import neuralnetwork.initialiser.UJMPFactory;
import neuralnetwork.inputs.NetworkInput;
import optimizers.StochasticGradientDescent;
import org.apache.log4j.BasicConfigurator;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class XORTester {

    private List<NetworkInput<OjAlgoMatrix>> data;

    private BufferedImage[] images;

    private double[][] xorData = new double[][] { { 0, 1 }, { 0, 0 }, { 1, 1 }, { 1, 0 } };
    private double[][] xorLabel = new double[][] { { 1 }, { 0 }, { 0 }, { 1 } };

    private NeuralNetwork<OjAlgoMatrix> network;
    private String path;

    private int imagesSize;
    private int w, h;

    XORTester(String path, int size) {
        BasicConfigurator.configure();
        this.path = path;
        this.imagesSize = size;
        w = 600;
        h = 600;
        images = new BufferedImage[imagesSize];

        data = new ArrayList<>();
        SecureRandom r = new SecureRandom();
        for (int i = 0; i < 10000; i++) {
            double[] cData;
            double[] cLabel;
            int rd = r.nextInt(xorData.length);
            cData = xorData[rd];
            cLabel = xorLabel[rd];
            data.add(new NetworkInput<OjAlgoMatrix>(new OjAlgoMatrix(cData, 2, 1), new OjAlgoMatrix(cLabel, 1, 1)));
        }
        Collections.shuffle(data);

        network = new NeuralNetwork<>(
                new NetworkBuilder<OjAlgoMatrix>(4).setFirstLayer(2).setLayer(3, new TanhFunction<>())
                        .setLayer(3, new TanhFunction<>()).setLastLayer(1, new TanhFunction<>())
                        .setCostFunction(new MeanSquaredCostFunction<>())
                        .setEvaluationFunction(new ThresholdEvaluationFunction<>(0.1))
                        .setOptimizer(new StochasticGradientDescent<>(0.6)),
                new OjAlgoFactory(new int[] { 2, 3, 3, 1 }, InitialisationMethod.XAVIER, InitialisationMethod.SCALAR));

        network.display();
        network.train(data.subList(0, 7000), data.subList(7000, 9000), 70, 64);
        // network.trainWithMetrics(data.subList(0, 1000), data.subList(1000, 2000), 70,
        // 64, "E:\\Programming\\Git\\JavaNeuralNetwork\\src\\main\\resources\\output");
    }

    public static void main(String[] args) throws IOException {
        new XORTester("C:\\Users\\edvin\\Downloads", 3);
    }

    private void run(int in) throws IOException {
        if (w % in != 0 && h % in != 0) {
            throw new IllegalArgumentException();
        }
        int cols = w / in;
        int rows = h / in;
        for (int l = 0; l < imagesSize; l++) {
            BufferedImage img = new BufferedImage(600, 600, BufferedImage.TYPE_INT_ARGB);
            images[l] = img;
            System.out.println("Image " + l);
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < rows; j++) {
                    double col = (double) i / cols;
                    double row = (double) j / rows;
                    double out = 0;
                    if (i % (in) == 0) {
                        out = network.predict(toInputMatrix(col, row)).sum();
                    }
                    out *= 255;
                    int colors = (int) out;
                    Color c = new Color(colors, colors, colors);
                    img.setRGB(i, j, c.getRGB());
                }
            }
        }

        int k = 0;
        for (BufferedImage img : images) {
            String out = path + "/XOR_" + k++ + ".png";
            ImageIO.write(img, "png", new File(out));
        }
    }

    private OjAlgoMatrix toInputMatrix(final double col, final double row) {
        return new OjAlgoMatrix(new double[][] { { col }, { row } }, 2, 1);
    }
}