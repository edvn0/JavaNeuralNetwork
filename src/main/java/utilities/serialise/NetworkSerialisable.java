package utilities.serialise;

import java.util.LinkedHashMap;

public interface NetworkSerialisable<K, V> {
    LinkedHashMap<K, V> params();
}
