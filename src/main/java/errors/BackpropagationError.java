package errors;

import java.util.InputMismatchException;

public class BackpropagationError extends InputMismatchException {

	/**
	 *
	 */
	private static final long serialVersionUID = -4577027705680874007L;

	public BackpropagationError(String s) {
		super(s);
	}
}
