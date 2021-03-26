package reinforcement.utils;

import lombok.Data;

@Data
public class EnvObservation {

	private final double[] observations;

	public EnvObservation(double... observations) {
		this.observations = observations;
	}

}
