package rl;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import reinforcement.games.SimpleGame;

public class TestSimpleGame {

	@Test
	public void testMoveSimpleGame() {
		SimpleGame g = new SimpleGame();
		g.reset();
		g.step(1);
		int pos = g.getPosition();
		assertEquals(pos, 1);

		g.step(1);
		pos = g.getPosition();
		assertEquals(pos, 2);
	}

	@Test
	public void testShouldThrowActionsOOR() {
		SimpleGame g = new SimpleGame();
		g.reset();
		var out = g.step(10);

		var s = out.getInfo().getInfo();

		assertEquals(s, "Invalid action.");
	}

	@Test
	public void testGame() {
		SimpleGame g = new SimpleGame();
		g.reset();
		g.render();
		boolean gameOver = false;
		while (!gameOver) {
			gameOver = g.isGameOver();
			g.render();
			g.step(g.getActionSpace().sample());
		}
	}

	@Test
	public void testRandomAgent() {
		SimpleGame g = new SimpleGame();
		double avgReward = 0.0f;
		double won = 0.0f;
		for (int i = 0; i < 100; i++) {
			g.reset();
			double reward = 0.0d;
			boolean gameOver = false;
			while (!gameOver) {
				gameOver = g.isGameOver();
				int randomAction = g.getActionSpace().sample();
				var obs = g.step(randomAction);
				reward += obs.getReward();
			}
			avgReward += reward;
			won += g.didWin() ? 1d : 0d;
		}
		System.out.println(avgReward / 100d);
		System.out.println(won / 100d);
	}

}
