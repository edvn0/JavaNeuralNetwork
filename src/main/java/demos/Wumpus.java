package demos;

import java.util.Scanner;
import reinforcement.games.WumpusGame;

public class Wumpus {

	public static void main(String[] args) {
		WumpusGame game = new WumpusGame(4);
		Scanner s = new Scanner(System.in);

		while (!game.gameOver()) {
			int action = Integer.parseInt(s.nextLine());
			game.step(action);
			game.render();
		}

	}

}
