# Noxim NoC simulation of SNN mappings

Simulates the spike traffic of a partitioned and placed SNN on a 64x64 mesh NoC with [Noxim](https://github.com/davidepatti/noxim).

## Setup

```sh
./setup_noxim.sh
```

This clones Noxim at a pinned commit into `noxim-src/`, applies `patches/`, fetches SystemC and yaml-cpp, and builds it.

## Run

```sh
./run_noxim.sh --snn <model.snn> --partition <mapping.part> --placement <mapping.plac> --run-name <name>
```

## Hardware model

- `config/mesh_64x64.yaml`: 64x64 wired mesh, 4-flit buffers, 1 VC, one spike = one 2-flit packet.
- `config/power.yaml`: 3.5 pJ per neighbor-wire traversal (1.75 pJ per flit) and 1.7 pJ per routing decision (charged once per packet per hop). All other costs are zero.