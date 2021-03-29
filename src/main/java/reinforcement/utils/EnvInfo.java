package reinforcement.utils;

import java.util.HashMap;
import java.util.Map;

public class EnvInfo {

	private Map<Integer, String> info;
	private int mapIndex;

	public EnvInfo() {
		this.info = new HashMap<>();
		this.mapIndex = 0;
	}

	public void addInfo(String info) {
		this.info.put(mapIndex++, info);
	}

	public String getInfo() {
		StringBuilder builder = new StringBuilder();
		for (var a : info.values()) {
			builder.append(a);
		}
		return builder.toString();
	}
}
