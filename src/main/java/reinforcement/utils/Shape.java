package reinforcement.utils;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Shape {

	private int X, Y, Z;

	public boolean contains(int val) {
		return X == val || Y == val || Z == val;
	}
}
