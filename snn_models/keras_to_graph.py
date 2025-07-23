# WARNING: RUNNING THIS FILE REQUIRES THE "tensorflow" PACKAGE, THAT IS NOT PRESENT IN "requirements.txt"!

from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Concatenate, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Input, Model
from tensorflow import keras
import tensorflow as tf

from typing import Optional

import networkx as nx
import numpy as np
import re
import os

"""
Extracts a directed neuron-level graph from a Keras model.
Nodes represent individual scalar activations, named as <layer_name>_coord.
Supported layers: InputLayer, Dense, Conv2D, ZeroPadding2D, DepthwiseConv2D, Add, Concatenate, Multiply, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Reshape, Dropout, Activation.
Batch size must be 1.
If 'directly_to_file' is provided, networkx is NOT used and the graph is directly written in graphml format on the provided path.
"""
def extract_neuron_graph(model : Model, use_layer_type_as_name : bool = False, directly_to_file : Optional[str] = None) -> Optional[nx.DiGraph]:
    print("Generating NN graph...")

    nodes = 0
    edges = 0
    if not directly_to_file:
        G = nx.DiGraph()
    else:
        G = open(directly_to_file, 'w')
        G.write(("<?xml version='1.0' encoding='utf-8'?>\n"
                 "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\""
                 "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\""
                 "xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns"
                 "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n"
                 "<graph edgedefault=\"directed\">\n"))

    layer_outputs = {} # id(layer) -> (shape, coordinate [valid neuron indices], node names, layer idx + name [when not creating nodes, this should be the name of the last layer that added nodes])
    layer_idx = 0 # increment only after handling layers that add nodes

    def add_node(name : str) -> None:
        nonlocal nodes
        nodes += 1
        if not directly_to_file:
            G.add_node(name)
        else:
            G.write(f"<node id=\"{name}\"/>\n")

    def add_edge(src : str, dst : str) -> None:
        nonlocal edges
        edges += 1
        if not directly_to_file:
            G.add_edge(src, dst)
        else:
            G.write(f"<edge source=\"{src}\" target=\"{dst}\"/>\n")

    for layer in model.layers:
        print(f"Working on layer {layer.name} (type {type(layer).__name__})...")
        # fix Keras's naming scheme that adds "_n" for layers with the same name after the first one,
        # thus screwing up the notation for dimensions; we move the index before the name...
        if use_layer_type_as_name:
            layer_name = f"{layer_idx}_{type(layer).__name__.lower()}"
        else:
            layer_name = f"{layer_idx}_{re.match(r'^((?:_?[a-zA-Z][a-zA-Z0-9]+)+)', layer.name).group(1)}"
        #layer_idx = len(layer_outputs)
        # get previous output mapping
        if isinstance(layer, tf.keras.layers.InputLayer):
            shape = layer.input_shape[0][1:]  # drop batch
            if not shape:
                print("WARNING: missing Input layer shape...")
                continue
            coords = list(np.ndindex(*shape))
            names = [f"{layer_name}_{'_'.join(map(str, coord))}" for coord in coords]
            for name in names:
                add_node(name)
            layer_outputs[id(layer)] = (shape, coords, names, layer_name)
            layer_idx += 1

        else:
            # identify inbound layer outputs
            inbound = []
            for node in layer._inbound_nodes:
                inbound_layer = node.inbound_layers
                if isinstance(inbound_layer, list):
                    inbound.extend(inbound_layer)
                else:
                    inbound.append(inbound_layer)
            if not inbound:
                print(f"WARNING: no inbound links on layer {layer.name}...")
                continue

            # for simplicity, assume single inbound tensor or two for Add
            inputs = []
            for in_layer in inbound:
                if id(in_layer) in layer_outputs:
                    inputs.append(layer_outputs[id(in_layer)])

            if not inputs or not inputs[0]:
                print(f"WARNING: no input tensors on layer {layer.name}...")
                continue

            # process supported layers
            # NOTE: either ensure that child classes are checked before parents, or switch from 'isinstance(c, t)' to 'type(c) == t' for strict matching!
            if isinstance(layer, tf.keras.layers.Dense):
                units = layer.units
                inp_shape, inp_coords, inp_names, in_name = inputs[0]
                out_coords = [(i,) for i in range(units)]
                out_names = [f"{layer_name}_{i}" for i in range(units)]
                for oname in out_names:
                    add_node(oname)
                for oname in out_names:
                    for iname in inp_names:
                        add_edge(iname, oname)
                layer_outputs[id(layer)] = ((units,), out_coords, out_names, layer_name)
                layer_idx += 1

            elif isinstance(layer, tf.keras.layers.ZeroPadding2D):
                inp_shape, inp_coords, inp_names, in_name = inputs[0]
                if len(inp_shape) != 3:
                    print(f"WARNING: input shape (\"{inp_shape}\") not of length 3 on layer {layer.name}...")
                    continue

                h_in, w_in, c = inp_shape
                pad = layer.padding
                if isinstance(pad, int):
                    pad = ((pad, pad), (pad, pad))  # symmetric
                (top, bottom), (left, right) = pad

                h_out = h_in + top + bottom
                w_out = w_in + left + right
                out_coords = list(np.ndindex(h_out, w_out, c))
                out_names = [f"{layer_name}_{i}_{j}_{k}" for (i, j, k) in out_coords]
                for name in out_names:
                    add_node(name)

                for (i, j, k), oname in zip(out_coords, out_names):
                    ii = i - top
                    jj = j - left
                    if 0 <= ii < h_in and 0 <= jj < w_in:
                        iname = f"{in_name}_{ii}_{jj}_{k}"
                        add_edge(iname, oname)

                layer_outputs[id(layer)] = ((h_out, w_out, c), out_coords, out_names, layer_name)
                layer_idx += 1

            elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                inp_shape, inp_coords, inp_names, in_name = inputs[0]
                if len(inp_shape) != 3:
                    print(f"WARNING: input shape (\"{inp_shape}\") not of length 3 on layer {layer.name}...")
                    continue

                h_in, w_in, c_in = inp_shape
                kh, kw = layer.kernel_size
                sh, sw = layer.strides
                padding = layer.padding
                depth_multiplier = layer.depth_multiplier

                if padding == 'same':
                    h_out = int(np.ceil(h_in / sh))
                    w_out = int(np.ceil(w_in / sw))
                    pad_h = max((h_out - 1) * sh + kh - h_in, 0)
                    pad_w = max((w_out - 1) * sw + kw - w_in, 0)
                else:  # valid
                    h_out = int(np.floor((h_in - kh + sh) / sh))
                    w_out = int(np.floor((w_in - kw + sw) / sw))
                    pad_h = 0
                    pad_w = 0

                c_out = c_in * depth_multiplier
                out_coords = list(np.ndindex(h_out, w_out, c_out))
                out_names = [f"{layer_name}_{i}_{j}_{k}" for i, j, k in out_coords]
                for oname in out_names:
                    add_node(oname)

                pad_top = pad_h // 2
                pad_left = pad_w // 2

                for idx, (i, j, k) in enumerate(out_coords):
                    out_name = out_names[idx]
                    in_c = k // depth_multiplier  # map output channel to corresponding input channel
                    for di in range(kh):
                        for dj in range(kw):
                            ii = i * sh + di - pad_top
                            jj = j * sw + dj - pad_left
                            if 0 <= ii < h_in and 0 <= jj < w_in:
                                in_name_full = f"{in_name}_{ii}_{jj}_{in_c}"
                                add_edge(in_name_full, out_name)

                layer_outputs[id(layer)] = ((h_out, w_out, c_out), out_coords, out_names, layer_name)
                layer_idx += 1

            elif isinstance(layer, tf.keras.layers.Conv2D):
                inp_shape, inp_coords, inp_names, in_name = inputs[0]
                if len(inp_shape) != 3:
                    print(f"WARNING: input shape (\"{inp_shape}\") not of length 3 on layer {layer.name}...")
                    continue
                h_in, w_in, c_in = inp_shape
                kh, kw = layer.kernel_size
                sh, sw = layer.strides
                padding = layer.padding

                if padding == 'same':
                    h_out = int(np.ceil(h_in / sh))
                    w_out = int(np.ceil(w_in / sw))
                    pad_h = max((h_out - 1) * sh + kh - h_in, 0)
                    pad_w = max((w_out - 1) * sw + kw - w_in, 0)
                else:  # valid
                    h_out = int(np.floor((h_in - kh + sh) / sh))
                    w_out = int(np.floor((w_in - kw + sw) / sw))
                    pad_h = 0
                    pad_w = 0

                c_out = layer.filters
                out_coords = list(np.ndindex(h_out, w_out, c_out))
                out_names = [f"{layer_name}_{i}_{j}_{k}" for i, j, k in out_coords]
                for oname in out_names:
                    add_node(oname)

                pad_top = pad_h // 2
                pad_left = pad_w // 2

                for idx, (i, j, k) in enumerate(out_coords):
                    out_name = out_names[idx]
                    for di in range(kh):
                        for dj in range(kw):
                            for cin in range(c_in):
                                ii = i * sh + di - pad_top
                                jj = j * sw + dj - pad_left
                                if 0 <= ii < h_in and 0 <= jj < w_in:
                                    iname = f"{in_name}_{ii}_{jj}_{cin}"
                                    add_edge(iname, out_name)
                layer_outputs[id(layer)] = ((h_out, w_out, c_out), out_coords, out_names, layer_name)
                layer_idx += 1

            elif isinstance(layer, tf.keras.layers.Add):
                # TODO: could remove new neurons from Add layers, and directly connect multiple edges into the subsequent layer
                inp1_shape, inp1_coords, inp1_names, inp1_name = inputs[0]
                inp2_shape, inp2_coords, inp2_names, inp2_name = inputs[1]
                coords = inp1_coords
                names = [f"{layer_name}_{'_'.join(map(str, coord))}" for coord in coords]
                for name in names:
                    add_node(name)
                for idx, coord in enumerate(coords):
                    out = names[idx]
                    add_edge(inp1_names[idx], out)
                    add_edge(inp2_names[idx], out)
                layer_outputs[id(layer)] = (inp1_shape, coords, names, layer_name)
                layer_idx += 1

            elif isinstance(layer, tf.keras.layers.Concatenate):
                axis = layer.axis - 1
                inp_shape, inp_coords, inp_names, in_name = inputs[0]
                merged_coords = []
                merged_names = []
                for _, coords, names, _ in inputs:
                    merged_coords.extend(coords)
                    merged_names.extend(names)
                out_shape = list(inputs[0][0])
                out_shape[axis] = sum(inp[0][axis] for inp in inputs)
                layer_outputs[id(layer)] = (tuple(out_shape), merged_coords, merged_names, in_name) # not adding node -> use the input layer's name!
                # NOTE: this "+= 1" is technically wrong, but exists to comply with SNN toolbox...
                layer_idx += 1

            elif isinstance(layer, tf.keras.layers.Multiply):
                # NOTE: ensure all input shapes match
                base_shape, base_coords, base_names, base_layer_name = inputs[0]
                coord_to_names = {coord: [] for coord in base_coords}

                for shape, coords, names, _ in inputs:
                    if shape != base_shape or len(coords) != len(base_coords):
                        print(f"WARNING: Shape mismatch in Multiply layer {layer.name}")
                        continue
                    for coord, name in zip(coords, names):
                        coord_to_names[coord].append(name)

                out_names = [f"{layer_name}_{'_'.join(map(str, coord))}" for coord in base_coords]
                for out_name in out_names:
                    add_node(out_name)

                for coord, out_name in zip(base_coords, out_names):
                    for in_name in coord_to_names[coord]:
                        add_edge(in_name, out_name)

                layer_outputs[id(layer)] = (base_shape, base_coords, out_names, layer_name)
                layer_idx += 1

            elif isinstance(layer, (tf.keras.layers.AveragePooling2D, tf.keras.layers.MaxPooling2D)):
                inp_shape, inp_coords, inp_names, in_name = inputs[0]
                h_in, w_in, c = inp_shape
                ph, pw = layer.pool_size
                sh, sw = layer.strides
                padding = layer.padding

                if padding == 'same':
                    h_out = int(np.ceil(h_in / sh))
                    w_out = int(np.ceil(w_in / sw))
                    pad_h = max((h_out - 1) * sh + ph - h_in, 0)
                    pad_w = max((w_out - 1) * sw + pw - w_in, 0)
                else:  # valid
                    h_out = int(np.floor((h_in - ph + sh) / sh))
                    w_out = int(np.floor((w_in - pw + sw) / sw))
                    pad_h = 0
                    pad_w = 0

                out_coords = list(np.ndindex(h_out, w_out, c))
                out_names = [f"{layer_name}_{i}_{j}_{k}" for i, j, k in out_coords]
                for oname in out_names:
                    add_node(oname)

                pad_top = pad_h // 2
                pad_left = pad_w // 2

                for idx, (i, j, k) in enumerate(out_coords):
                    out_name = out_names[idx]
                    for di in range(ph):
                        for dj in range(pw):
                            ii = i * sh + di - pad_top
                            jj = j * sw + dj - pad_left
                            if 0 <= ii < h_in and 0 <= jj < w_in:
                                iname = f"{in_name}_{ii}_{jj}_{k}"
                                add_edge(iname, out_name)
                layer_outputs[id(layer)] = ((h_out, w_out, c), out_coords, out_names, layer_name)
                layer_idx += 1

            elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                inp_shape, inp_coords, inp_names, in_name = inputs[0]
                if len(inp_shape) != 3:
                    print(f"WARNING: input shape (\"{inp_shape}\") not of length 3 on layer {layer.name}...")
                    continue

                h, w, c = inp_shape
                out_coords = [(0, 0, i,) for i in range(c)]
                # NOTE: for compatibility with SNN toolbox, we gotta keep the width and height idxs...
                out_names = [f"{layer_name}_0_0_{i}" for i in range(c)]
                for name in out_names:
                    add_node(name)

                for i in range(c):
                    out_name = f"{layer_name}_0_0_{i}"
                    for h_idx in range(h):
                        for w_idx in range(w):
                            in_name_full = f"{in_name}_{h_idx}_{w_idx}_{i}"
                            add_edge(in_name_full, out_name)

                layer_outputs[id(layer)] = ((c,), out_coords, out_names, layer_name)
                layer_idx += 1

            elif isinstance(layer, tf.keras.layers.Flatten):
                inp_shape, inp_coords, inp_names, in_name = inputs[0]
                layer_outputs[id(layer)] = ((np.prod(inp_shape),), [(i,) for i in range(len(inp_names))], inp_names, in_name) # not adding node -> use the input layer's name!
                # NOTE: this "+= 1" is technically wrong, but exists to comply with SNN toolbox...
                layer_idx += 1

            elif isinstance(layer, tf.keras.layers.Reshape):
                inp_shape, inp_coords, inp_names, in_name = inputs[0]
                new_shape = layer.target_shape
                if np.prod(inp_shape) != np.prod(new_shape):
                    print(f"WARNING: reshape mismatch in layer {layer.name}: {inp_shape} -> {new_shape}...")

                layer_outputs[id(layer)] = (new_shape, list(np.ndindex(*new_shape)), inp_names, in_name)  # Keep same node names
                # NOTE: this "+= 1" is technically wrong, but exists to comply with SNN toolbox...
                layer_idx += 1

            elif isinstance(layer, tf.keras.layers.Dropout):
                if len(inputs) != 1:
                    print(f"WARNING: while stitching together input-to-output of layer {layer.name}, the layer had multiple inputs...")
                layer_outputs[id(layer)] = inputs[0]
                # NOTE: this "+= 1" is technically wrong, but exists to comply with SNN toolbox...
                layer_idx += 1

            elif isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Activation, tf.keras.layers.ReLU)):
                if len(inputs) != 1:
                    print(f"WARNING: while stitching together input-to-output of layer {layer.name}, the layer had multiple inputs...")
                layer_outputs[id(layer)] = inputs[0]

            else:
                print(f"WARNING: unrecognized layer type ({type(layer)}) for layer {layer.name}...")
                continue
        
        print("Current graph size:", nodes, "nodes", edges, "edges...")

    print("Graph generation complete!")
    if directly_to_file:
        G.write("</graph></graphml>")
        G.close()
    return G

# Example usage:
if __name__ == "__main__":
    path_wd = os.getcwd()

    # load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize
    x_train = x_train / 255
    x_test = x_test / 255
    # add channel dimension (B&W => 1 channel)
    axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    x_train = np.expand_dims(x_train, axis)
    x_test = np.expand_dims(x_test, axis)
    # one-hot encode target vectors
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # create ANN
    input_shape = x_train.shape[1:]
    input_layer = Input(input_shape)
    layer = Conv2D(filters = 16, kernel_size = (5, 5), strides = (2, 2), activation = 'relu')(input_layer)
    layer = BatchNormalization(axis = axis)(layer)
    layer = Activation('relu')(layer)
    layer = AveragePooling2D()(layer)
    branch1 = Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu')(layer)
    branch2 = Conv2D(filters = 8, kernel_size = (1, 1), activation = 'relu')(layer)
    layer = Concatenate(axis = axis)([branch1, branch2])
    layer = Conv2D(filters = 10, kernel_size = (3, 3), activation = 'relu')(layer)
    layer = Flatten()(layer)
    layer = Dropout(0.01)(layer)
    layer = Dense(units = 10, activation = 'softmax')(layer)

    model = Model(input_layer, layer)
    model.summary()
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    # train model
    model.fit(x_train, y_train, batch_size = 64, epochs = 1, verbose = 2, validation_data = (x_test, y_test))

    # save the model as both a NN and a graph
    model_name = 'mnist_cnn'
    keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))
    nx.write_graphml(extract_neuron_graph(model), os.path.join(path_wd, model_name + '.graphml'))