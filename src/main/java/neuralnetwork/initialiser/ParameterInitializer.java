package neuralnetwork.initialiser;

import java.util.List;
import math.linearalgebra.Matrix;
import math.linearalgebra.simple.SMatrix;
import org.ojalgo.matrix.Primitive64Matrix;
import utilities.types.Pair;

public abstract class ParameterInitializer<M> {

	protected int[] sizes;
	protected InitialisationMethod wM;
	protected InitialisationMethod bM;

	public ParameterInitializer(InitialisationMethod weightMethod,
		InitialisationMethod biasMethod) {
		this.wM = weightMethod;
		this.bM = biasMethod;
	}

	public static ParameterInitializer<?> get(InitialisationMethod wM, InitialisationMethod bM,
		String name,
		Class<?> typeOf) {
		if (typeOf.equals(SMatrix.class)) {
			return new SimpleInitializer(wM, bM);
		} else if (typeOf.equals(Primitive64Matrix.class)) {
			return new OjAlgoInitializer(wM, bM);
		} else if (typeOf.equals(org.ujmp.core.Matrix.class)) {
			return new UJMPInitializer(wM, bM);
		} else {
			throw new IllegalArgumentException("Unsupported initializer for this type.");
		}
	}

	public void init(int[] sizes) {
		this.sizes = sizes;
	}

	public abstract List<Matrix<M>> getWeightParameters();

	public abstract List<Matrix<M>> getBiasParameters();

	public List<Matrix<M>> getDeltaWeightParameters() {
		return getDeltaParameters(false);
	}

	protected abstract List<Matrix<M>> getDeltaParameters(boolean isBias);

	public List<Matrix<M>> getDeltaBiasParameters() {
		return getDeltaParameters(true);
	}

	public abstract Matrix<M> getFirstBias();

	public abstract String name();

	public Pair<InitialisationMethod, InitialisationMethod> getMethods() {
		return Pair.of(wM, bM);
	}

}
