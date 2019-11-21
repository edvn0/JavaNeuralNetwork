package errors;

import java.util.InputMismatchException;

public class BackpropagationError extends InputMismatchException {

	public BackpropagationError(String s) {
		super(s);
	}
}
