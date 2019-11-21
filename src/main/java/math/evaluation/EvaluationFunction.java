package math.evaluation;

import java.util.List;
import matrix.Matrix;

public interface EvaluationFunction {

	Matrix evaluatePrediction(List<Matrix[]> toEvaluate);

}
