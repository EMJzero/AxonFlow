This folder contains a set of benchmark hypergraphs that have been preprocessed and are ready for use.

From any python script, to import an hypergraph you can use a function like this one:
```python
def load(path : str) -> tuple[int, tuple[int, tuple[int], float]]:
    import struct
    with open(path, 'rb') as f:
        nodes, num_edges = struct.unpack('<II', f.read(8))
        hyperedges = [] # format: (source node, destination nodes tuple, spike frequency)
        for _ in range(num_edges):
            num_dsts = struct.unpack('<I', f.read(4))[0]
            src = struct.unpack('<I', f.read(4))[0]
            dsts = array('I')
            dsts.frombytes(f.read(4 * num_dsts))
            freq = struct.unpack('<f', f.read(4))[0]
            hyperedges.append((src, dsts, freq))
    return nodes, hyperedges
```

Viceversa, those hypergraphs have been exported via a function like:
```python
def save(path : str, nodes : int, hyperedges : tuple[int, tuple[int], float]) -> None:
    import struct
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', nodes, len(hyperedges)))
        for src, dsts, freq in hyperedges:
            f.write(struct.pack('<I', len(dsts)))
            f.write(struct.pack('<I', src))
            f.write(array('I', dsts).tobytes())
            f.write(struct.pack('<f', freq))
```

Analogous implementations of those functions can be found in [snn.py](../snn.py).