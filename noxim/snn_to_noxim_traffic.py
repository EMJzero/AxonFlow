from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Iterable


def expand_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def read_exact(file: BinaryIO, size: int, what: str) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected end of file while reading {what}")
    return data


def read_int_list(path: Path) -> list[int]:
    data = path.read_bytes()
    item_size = struct.calcsize("i")
    if len(data) % item_size != 0:
        raise ValueError(f"{path} size is not a multiple of {item_size} bytes")
    if not data:
        return []
    return list(struct.unpack(f"{len(data) // item_size}i", data))


def read_partition(path: Path) -> list[int]:
    partitions = read_int_list(path)
    if not partitions:
        raise ValueError("Partition file is empty")
    min_part = min(partitions)
    if min_part < 0:
        raise ValueError(f"Partition file contains a negative partition id: {min_part}")
    return partitions


def read_placement(path: Path) -> list[tuple[int, int]]:
    values = read_int_list(path)
    if len(values) % 2 != 0:
        raise ValueError("Placement file contains an odd number of integers")
    coords = list(zip(values[::2], values[1::2]))
    if not coords:
        raise ValueError("Placement file is empty")
    return coords


def tile_id(coord: tuple[int, int], mesh_dim_x: int, mesh_dim_y: int) -> int:
    x, y = coord
    if x < 0 or y < 0 or x >= mesh_dim_x or y >= mesh_dim_y:
        raise ValueError(
            f"Placement coordinate {(x, y)} is outside the {mesh_dim_x}x{mesh_dim_y} mesh"
        )
    return y * mesh_dim_x + x


def make_partition_to_tile(
    placement: list[tuple[int, int]], mesh_dim_x: int, mesh_dim_y: int
) -> list[int]:
    return [tile_id(coord, mesh_dim_x, mesh_dim_y) for coord in placement]


def iter_snn_hyperedges(path: Path) -> Iterable[tuple[int, tuple[int, ...], float]]:
    with path.open("rb") as file:
        nodes, num_hyperedges = struct.unpack("<II", read_exact(file, 8, "SNN header"))
        for edge_idx in range(num_hyperedges):
            num_dsts = struct.unpack("<I", read_exact(file, 4, f"edge {edge_idx} destination count"))[0]
            src = struct.unpack("<I", read_exact(file, 4, f"edge {edge_idx} source"))[0]
            dsts = struct.unpack(
                f"<{num_dsts}I",
                read_exact(file, 4 * num_dsts, f"edge {edge_idx} destinations"),
            )
            freq = struct.unpack("<f", read_exact(file, 4, f"edge {edge_idx} spike frequency"))[0]

            if src >= nodes:
                raise ValueError(f"Edge {edge_idx} has source {src}, but SNN has {nodes} nodes")
            bad_dsts = [dst for dst in dsts if dst >= nodes]
            if bad_dsts:
                raise ValueError(
                    f"Edge {edge_idx} contains destination {bad_dsts[0]}, but SNN has {nodes} nodes"
                )
            yield src, dsts, freq

        leftover = file.read(1)
        if leftover:
            raise ValueError(f"{path} contains trailing bytes after {num_hyperedges} hyperedges")


def read_snn_node_count(path: Path) -> int:
    with path.open("rb") as file:
        nodes, _ = struct.unpack("<II", read_exact(file, 8, "SNN header"))
    return nodes


def aggregate_traffic(
    snn_path: Path,
    neuron_to_partition: list[int],
    partition_to_tile: list[int],
    include_local: bool,
    fanout_mode: str,
    row_granularity: str,
) -> tuple[list[tuple[int, int, float]], dict[str, int | float]]:
    traffic: defaultdict[tuple[int, ...], float] = defaultdict(float)
    stats: dict[str, int | float] = {
        "hyperedges": 0,
        "destinations": 0,
        "zero_frequency_hyperedges": 0,
        "local_packets_skipped": 0,
        "negative_frequency_hyperedges": 0,
    }

    for src, dsts, freq in iter_snn_hyperedges(snn_path):
        stats["hyperedges"] += 1
        stats["destinations"] += len(dsts)

        if freq == 0.0:
            stats["zero_frequency_hyperedges"] += 1
            continue
        if freq < 0.0:
            stats["negative_frequency_hyperedges"] += 1
            continue

        src_tile = partition_to_tile[neuron_to_partition[src]]

        if fanout_mode == "per-destination-core":
            dst_tiles = {
                partition_to_tile[neuron_to_partition[dst]]
                for dst in dsts
            }
            for dst_tile in dst_tiles:
                if src_tile == dst_tile and not include_local:
                    stats["local_packets_skipped"] += 1
                    continue
                if row_granularity == "source-neuron-destination-core":
                    traffic[(src, src_tile, dst_tile)] += freq
                else:
                    traffic[(src_tile, dst_tile)] += freq
        else:
            for dst in dsts:
                dst_tile = partition_to_tile[neuron_to_partition[dst]]
                if src_tile == dst_tile and not include_local:
                    stats["local_packets_skipped"] += 1
                    continue
                if row_granularity == "source-neuron-destination-core":
                    traffic[(src, src_tile, dst_tile)] += freq
                else:
                    traffic[(src_tile, dst_tile)] += freq

    rows = []
    for key, packets_per_second in traffic.items():
        if row_granularity == "source-neuron-destination-core":
            _, src_tile, dst_tile = key
        else:
            src_tile, dst_tile = key
        rows.append((src_tile, dst_tile, packets_per_second))

    stats["traffic_rows"] = len(rows)
    stats["unique_tile_pairs"] = len({(src, dst) for src, dst, _ in rows})
    stats["packets_per_second"] = sum(traffic.values())
    return rows, stats


def write_noxim_traffic_table(
    output_path: Path,
    traffic_rows: list[tuple[int, int, float]],
    noc_clock_hz: float,
    precision: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as file:
        file.write("% src dst pir por\n")
        for src, dst, packets_per_second in sorted(traffic_rows):
            pir = packets_per_second / noc_clock_hz
            file.write(f"{src} {dst} {pir:.{precision}g} {pir:.{precision}g}\n")


def validate_mapping(
    snn_path: Path,
    neuron_to_partition: list[int],
    partition_to_tile: list[int],
) -> None:
    snn_nodes = read_snn_node_count(snn_path)
    if len(neuron_to_partition) != snn_nodes:
        raise ValueError(
            f"Partition file contains {len(neuron_to_partition)} assignments, "
            f"but SNN contains {snn_nodes} nodes"
        )

    max_partition = max(neuron_to_partition)
    if max_partition >= len(partition_to_tile):
        raise ValueError(
            f"Partition file references partition {max_partition}, "
            f"but placement file contains only {len(partition_to_tile)} coordinates"
        )


def source_pirs(
    traffic_rows: list[tuple[int, int, float]], noc_clock_hz: float
) -> dict[int, float]:
    per_source_pir: defaultdict[int, float] = defaultdict(float)
    for src, _, packets_per_second in traffic_rows:
        per_source_pir[src] += packets_per_second / noc_clock_hz
    return per_source_pir


def print_summary(
    output_path: Path,
    traffic_rows: list[tuple[int, int, float]],
    stats: dict[str, int | float],
    noc_clock_hz: float,
) -> None:
    per_source_pir = source_pirs(traffic_rows, noc_clock_hz)
    saturated_sources = [src for src, pir in per_source_pir.items() if pir > 1.0]
    max_source_pir = max(per_source_pir.values(), default=0.0)

    print(f"Wrote {output_path}")
    print(f"Traffic rows: {stats['traffic_rows']}")
    print(f"Unique tile pairs: {stats['unique_tile_pairs']}")
    print(f"Aggregate packets/s from SNN spike frequencies: {stats['packets_per_second']:.6g}")
    print(f"NoC clock: {noc_clock_hz:.6g} Hz")
    print(f"Max aggregate PIR from one source tile: {max_source_pir:.6g}")
    print(f"Local packets skipped: {stats['local_packets_skipped']}")
    if stats["negative_frequency_hyperedges"]:
        print(f"Warning: skipped {stats['negative_frequency_hyperedges']} negative-frequency hyperedges")
    if saturated_sources:
        print(
            "Warning: "
            f"{len(saturated_sources)} source tile(s) have cumulative PIR > 1; "
            "Noxim table traffic samples at most one new packet per source per cycle."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Axon SNN, partitioning, and placement files into a Noxim traffic table."
    )
    parser.add_argument("--snn", required=True, help="Path to the binary .snn file")
    parser.add_argument("--partition", required=True, help="Path to the binary .part file")
    parser.add_argument("--placement", required=True, help="Path to the binary .plac file")
    parser.add_argument("--output", required=True, help="Output Noxim traffic table path")
    parser.add_argument("--mesh-dim-x", type=int, default=64, help="Noxim mesh width")
    parser.add_argument("--mesh-dim-y", type=int, default=64, help="Noxim mesh height")
    parser.add_argument(
        "--clock-period-ps",
        type=float,
        default=100_000.0,
        help=(
            "Noxim clock period in picoseconds. SNN spike frequencies are packets/s, "
            "and PIR is computed as packets/s divided by the resulting NoC clock Hz."
        ),
    )
    parser.add_argument(
        "--fanout-mode",
        choices=("per-destination-core", "per-destination-neuron"),
        default="per-destination-core",
        help=(
            "How to count a spike whose destinations include multiple neurons on the same core. "
            "'per-destination-core' emits one packet per destination tile; "
            "'per-destination-neuron' emits one packet per destination neuron."
        ),
    )
    parser.add_argument(
        "--row-granularity",
        choices=("tile-pair", "source-neuron-destination-core"),
        default="tile-pair",
        help=(
            "How much to collapse rows in the traffic table. 'tile-pair' writes one row per "
            "source/destination tile pair. 'source-neuron-destination-core' preserves original "
            "source-neuron granularity while still writing Noxim tile IDs."
        ),
    )
    parser.add_argument(
        "--no-merge-traffic",
        action="store_true",
        help=(
            "Shortcut for --row-granularity source-neuron-destination-core. "
            "Useful for auditing, but much slower in Noxim."
        ),
    )
    parser.add_argument(
        "--meta-output",
        help=(
            "Optional JSON file receiving the mesh dimensions and traffic statistics, "
            "for consumption by run scripts"
        ),
    )
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Keep traffic whose source and destination map to the same Noxim tile",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=12,
        help="Significant digits used when writing PIR/POR values",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_merge_traffic:
        args.row_granularity = "source-neuron-destination-core"

    if args.mesh_dim_x <= 0 or args.mesh_dim_y <= 0:
        raise ValueError("Mesh dimensions must be positive")
    if args.clock_period_ps <= 0.0:
        raise ValueError("Clock period must be positive")
    if args.precision <= 0:
        raise ValueError("Precision must be positive")

    snn_path = expand_path(args.snn)
    partition_path = expand_path(args.partition)
    placement_path = expand_path(args.placement)
    output_path = expand_path(args.output)

    neuron_to_partition = read_partition(partition_path)
    placement = read_placement(placement_path)
    mesh_dim_x, mesh_dim_y = args.mesh_dim_x, args.mesh_dim_y
    partition_to_tile = make_partition_to_tile(placement, mesh_dim_x, mesh_dim_y)
    validate_mapping(snn_path, neuron_to_partition, partition_to_tile)

    traffic_rows, stats = aggregate_traffic(
        snn_path=snn_path,
        neuron_to_partition=neuron_to_partition,
        partition_to_tile=partition_to_tile,
        include_local=args.include_local,
        fanout_mode=args.fanout_mode,
        row_granularity=args.row_granularity,
    )

    noc_clock_hz = 1.0 / (args.clock_period_ps * 1e-12)
    write_noxim_traffic_table(output_path, traffic_rows, noc_clock_hz, args.precision)
    print_summary(output_path, traffic_rows, stats, noc_clock_hz)

    if args.meta_output:
        per_source_pir = source_pirs(traffic_rows, noc_clock_hz)
        total_pps = sum(pps for _, _, pps in traffic_rows)
        hop_weighted_pps = sum(
            pps * (abs(src % mesh_dim_x - dst % mesh_dim_x) + abs(src // mesh_dim_x - dst // mesh_dim_x))
            for src, dst, pps in traffic_rows
        )
        meta = {
            "mesh_dim_x": mesh_dim_x,
            "mesh_dim_y": mesh_dim_y,
            "used_tiles": len(set(partition_to_tile)),
            "noc_clock_hz": noc_clock_hz,
            "fanout_mode": args.fanout_mode,
            "expected_packets_per_second": total_pps,
            "expected_average_hops": hop_weighted_pps / total_pps if total_pps else 0.0,
            "max_source_pir": max(per_source_pir.values(), default=0.0),
            "saturated_source_tiles": sum(1 for pir in per_source_pir.values() if pir > 1.0),
            **stats,
        }
        meta_path = expand_path(args.meta_output)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
