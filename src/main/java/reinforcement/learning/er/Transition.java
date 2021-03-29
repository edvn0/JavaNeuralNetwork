package reinforcement.learning.er;

import lombok.AllArgsConstructor;
import lombok.Data;
import reinforcement.utils.EnvObservation;

@Data
@AllArgsConstructor
public class Transition {

	EnvObservation s;
	int action;
	double reward;
	EnvObservation newS;
	boolean done;

	public Transition(EnvObservation s, int action, double reward, EnvObservation newS, int done) {
		this(s, action, reward, newS, done == 0);
	}

}
