# AxonFlow

Author: Ronzani Marco @EMJzero

## Setup

Environment: **Linux, e.g. Ubuntu 22.04**

Required: **python >= 3.10**<br>
Recommended: **python == 3.13**

```sh
pip install -r requirements.txt
```

## Usage

Run `main.py` to try all mapping algorithm combinations at once on a given network.
> Deprecated: use `targeted_main.py` for this.

Run `main.py -d` to access the SNN import/export/save functionalities without running any mapping algorithm.
> TODO: create a dedicated main file for this.

Run `scalability_main.py` to test all mapping algorithm across increasingly large randomly generated hypergraphs.

Run `targeted_main.py` to test all mapping algorithms on a specific imported hypergraph.

Tweak `settings.py` to set logging and multiprocessing related settings.

To generate some plots, consider using, under `/plots`, `all_in_one_bars.py`, `targeted_bars.py`, `partitioning_bars.py`, and `manual_spikefrequency_distribution.py`.

## Handling Hypergraphs

Refer to [`load_store.py`](load_store.py) for a few words on how to generate SNNs from ANNs using [SNN toolbox](https://github.com/NeuromorphicProcessorProject/snn_toolbox) and then extract their hypergraphs.
A few words on how to prepare the Allen V1 model are also in there.
> TODO: write a dedicated guide.md for that!

When you run any main file, you can save the hypergraph it will work on via the `-s <path>` option, viceversa you can let any main run while using a previously stored hypergraph through the `-r <path>` option.

## Concepts

|              | **cost** | **hardware**       | **mapping**          | **hypergraph**     |
|--------------|----------|--------------------|----------------------|--------------------|
| partitioning | cuts     | spike multicast    | synaptic reuse       | 2nd order affinity |
| layout       | hops     | manhattan distance | connections locality | 1st order affinity |