package utilities.serialise;

import java.util.LinkedHashMap;

public interface NetworkSerializable<K, V> {
    LinkedHashMap<K, V> params();
}
