package reinforcement.utils;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Sars {

	private EnvObservation observation;
	private double reward;
	private boolean done;
	private EnvInfo info;
}
