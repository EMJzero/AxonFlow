from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True, help="Noxim JSON stats file")
    parser.add_argument("--traffic-meta", required=True, help="Metadata from snn_to_noxim_traffic.py")
    parser.add_argument("--clock-period-ps", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=0, help="Warmup cycles excluded from stats")
    parser.add_argument("--routing", default="", help="Routing algorithm, recorded in the summary")
    parser.add_argument("--output", help="Write the summary as JSON here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = json.loads(Path(args.stats).read_text())
    meta = json.loads(Path(args.traffic_meta).read_text())
    config, summary = stats["config"], stats["summary"]

    period_s = args.clock_period_ps * 1e-12
    measured_cycles = config["simulation_time_cycles"] - args.warmup
    measured_s = measured_cycles * period_s

    received = summary["total_received_packets"]
    expected_pps = meta["expected_packets_per_second"]
    received_pps = received / measured_s if measured_s > 0 else 0.0
    delivery_ratio = received_pps / expected_pps if expected_pps > 0 else float("nan")

    total_energy = summary["total_energy_j"]
    dynamic_energy = summary["dynamic_energy_j"]

    result = {
        "routing": args.routing,
        "mesh_dim_x": config["mesh_dim_x"],
        "mesh_dim_y": config["mesh_dim_y"],
        "used_tiles": meta["used_tiles"],
        "noc_clock_hz": 1.0 / period_s,
        "simulated_seconds": measured_s,
        "simulated_cycles": measured_cycles,
        "expected_packets_per_second": expected_pps,
        "received_packets": received,
        "received_packets_per_second": received_pps,
        "delivery_ratio": delivery_ratio,
        "expected_average_hops": meta["expected_average_hops"],
        "average_latency_cycles": summary["global_average_delay_cycles"],
        "average_latency_s": summary["global_average_delay_cycles"] * period_s,
        "total_energy_j": total_energy,
        "dynamic_energy_j": dynamic_energy,
        "average_power_w": total_energy / measured_s if measured_s > 0 else 0.0,
        "energy_per_packet_j": dynamic_energy / received if received else float("nan"),
        "max_source_pir": meta["max_source_pir"]
    }

    warnings = []
    if meta["saturated_source_tiles"]:
        warnings.append(
            f"{meta['saturated_source_tiles']} source tile(s) need more than one packet per cycle; "
            "Noxim injects at most one, so traffic is under-represented. Raise the NoC frequency."
        )
    if expected_pps * measured_s >= 100 and delivery_ratio < 0.9:
        warnings.append(
            f"Only {delivery_ratio:.1%} of the expected spike traffic was delivered: "
            "the network is saturated or deadlocked, latency/energy are not meaningful."
        )
    if expected_pps * measured_s < 100:
        warnings.append(
            f"Only ~{expected_pps * measured_s:.0f} packets expected in the simulated window; "
            "statistics are noisy. Simulate longer for stable numbers."
        )
    result["warnings"] = warnings

    print()
    print("=== Physical summary ===")
    print(f"Mesh: {result['mesh_dim_x']}x{result['mesh_dim_y']} ({result['used_tiles']} tiles used), routing {args.routing}")
    print(f"Simulated: {measured_s:.6g} s ({measured_cycles} cycles at {result['noc_clock_hz']:.6g} Hz)")
    print(f"Spike packets/s: expected {expected_pps:.6g}, delivered {received_pps:.6g} ({delivery_ratio:.1%})")
    print(f"Average hops: {result['expected_average_hops']:.3f}")
    print(
        f"Latency: avg {result['average_latency_cycles']:.3f} cycles ({result['average_latency_s']:.6g} s), "
        f"max {result['max_latency_cycles']} cycles ({result['max_latency_s']:.6g} s)"
    )
    print(f"Average NoC power: {result['average_power_w']:.6g} W")
    print(f"Energy per spike packet: {result['energy_per_packet_j'] * 1e12:.4g} pJ")
    for warning in warnings:
        print(f"Warning: {warning}")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
