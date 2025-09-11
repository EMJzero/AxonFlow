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

| ------------ | **cost** | **hardware**       | **mapping**          | **hypergraph**     |
|--------------|----------|--------------------|----------------------|--------------------|
| partitioning | cuts     | spike multicast    | synaptic reuse       | 2nd order affinity |
| layout       | hops     | manhattan distance | connections locality | 1st order affinity |

#### Defined as:
- **synaptic reuse** occurs when multiple neurons in the same partition, and therefore hardware core, receive spikes from a common source (axon).
- **connections locality** occurs when many hedges can be fully resolved within a small neighborhood of cores.
- **First-order affinity**, the weight of the direct connection between a source and a destination node.
- **Second-order affinity**, the weight of co-membership, that is, the cumulative weight of the hedges a set of nodes partakes in together.

#### Summarizing:
- to reduce **cuts** during **partitioning**, a mapping exploits NMH's **multicast** capabilities through **synaptic reuse** by grouping nodes with high **second-order affinity**.
- to limit spike **hops** after **layout**, via shorter **Manhattan distances** on NMH, a mapping increases **connections locality** by pulling closer partitions with high **first-order affinity**.

> The term "affinity" was used instead of "adjacency" because the latter only implies the existence of a connection to move between nodes, while the former hints at the idea of "we are similar, so let's be together".

#### Why hypergraphs:
<img src="static/houkago_kitaku_biyori_hyper.png" width="250px" padding-left="15px" align="right"/>
In graph form, a SNN’s source node connects to multiple destinations with a different edge for each, thus losing access to the notion of second-order affinity.
Practically, this hinders the quantification and realization of synaptic reuse, where you need to reason in terms of the common connections between nodes.
With hypergraphs, instead, hyperedges naturally group together nodes under a joint source.

> In short, merely using a graph would make synaptic reuse invisible and leads to overestimation of communication costs.


## TODOS BEFORE RELEASING THIS CODE:

- add support for "synapses per core" constraint