#!/usr/bin/env bash
set -euo pipefail

HERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NOXIM_DIR="${NOXIM_DIR:-${HERE}/noxim-src}"

SNN="${SNN:-}"
PARTITION="${PARTITION:-}"
PLACEMENT="${PLACEMENT:-}"

CONFIG="${CONFIG:-${HERE}/config/mesh_64x64.yaml}"
POWER="${POWER:-${HERE}/config/power.yaml}"
RESULTS_DIR="${RESULTS_DIR:-${HERE}/results}"
RUN_NAME="${RUN_NAME:-run}"

FREQUENCY_HZ="${FREQUENCY_HZ:-1000000}"
SIM_SECONDS="${SIM_SECONDS:-1}"
SIM_CYCLES="${SIM_CYCLES:-}"
WARMUP_CYCLES="${WARMUP_CYCLES:-0}"
SEED="${SEED:-0}"
ROUTING="${ROUTING:-XY}"
FANOUT_MODE="${FANOUT_MODE:-per-destination-core}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10000}"

MESH_DIM_X=64
MESH_DIM_Y=64

usage() {
  cat <<EOF
Usage: $(basename "$0") --snn PATH --partition PATH --placement PATH [options]

Converts an SNN mapping into a Noxim traffic table and simulates it on the
64x64 mesh described by config/. Run setup_noxim.sh once beforehand.

Options:
  --snn PATH              Binary .snn file (required)
  --partition PATH        Binary .part file (required)
  --placement PATH        Binary .plac file (required)
  --results-dir PATH      Output directory. Default: ${RESULTS_DIR}
  --run-name NAME         Output filename prefix. Default: ${RUN_NAME}
  --frequency-hz HZ       NoC frequency. Default: ${FREQUENCY_HZ}
  --sim-seconds SECONDS   Simulated seconds. Default: ${SIM_SECONDS}
  --sim CYCLES            Override simulation cycles. Default: frequency * seconds
  --warmup CYCLES         Warmup cycles. Default: ${WARMUP_CYCLES}
  --seed N                Noxim RNG seed. Default: ${SEED}
  --routing ALGO          XY, NORTH_LAST, ODD_EVEN, WEST_FIRST, NEGATIVE_FIRST.
                          Default: ${ROUTING}
  --fanout-mode MODE      per-destination-core or per-destination-neuron.
                          Default: ${FANOUT_MODE}
  --progress-interval N   Print progress every N cycles, 0 to disable.
                          Default: ${PROGRESS_INTERVAL}
  -h, --help              Show this help

Options can also be set through same-name uppercase environment variables.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snn) SNN="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --placement) PLACEMENT="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --frequency-hz) FREQUENCY_HZ="$2"; shift 2 ;;
    --sim-seconds) SIM_SECONDS="$2"; shift 2 ;;
    --sim) SIM_CYCLES="$2"; shift 2 ;;
    --warmup) WARMUP_CYCLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --routing) ROUTING="$2"; shift 2 ;;
    --fanout-mode) FANOUT_MODE="$2"; shift 2 ;;
    --progress-interval) PROGRESS_INTERVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for var in SNN PARTITION PLACEMENT; do
  if [[ -z "${!var}" ]]; then
    echo "Missing --$(tr '[:upper:]' '[:lower:]' <<< "${var}")" >&2
    usage >&2
    exit 2
  fi
done

if [[ ! -x "${NOXIM_DIR}/bin/noxim" ]]; then
  echo "Noxim not found in ${NOXIM_DIR}, run ${HERE}/setup_noxim.sh first" >&2
  exit 1
fi

case "${ROUTING}" in
  XY|NORTH_LAST|ODD_EVEN|WEST_FIRST|NEGATIVE_FIRST) ;;
  *) echo "Unsupported routing: ${ROUTING}" >&2; exit 2 ;;
esac

mkdir -p "${RESULTS_DIR}"
TRAFFIC="${RESULTS_DIR}/${RUN_NAME}.traffic.txt"
TRAFFIC_META="${RESULTS_DIR}/${RUN_NAME}.traffic_meta.json"
RUN_CONFIG="${RESULTS_DIR}/${RUN_NAME}.config.yaml"
RESULT_JSON="${RESULTS_DIR}/${RUN_NAME}.json"
RESULT_TXT="${RESULTS_DIR}/${RUN_NAME}.txt"
SUMMARY_JSON="${RESULTS_DIR}/${RUN_NAME}.summary.json"

read -r CLOCK_PERIOD_PS SIM_CYCLES < <(
  python3 - "${FREQUENCY_HZ}" "${SIM_SECONDS}" "${SIM_CYCLES}" <<'PY'
import sys
from decimal import Decimal, ROUND_HALF_UP

frequency_hz, sim_seconds, sim_cycles = Decimal(sys.argv[1]), Decimal(sys.argv[2]), sys.argv[3]
if frequency_hz <= 0 or sim_seconds <= 0:
    raise SystemExit("frequency and simulated seconds must be positive")
period = (Decimal(10) ** 12 / frequency_hz).to_integral_value(rounding=ROUND_HALF_UP)
cycles = Decimal(sim_cycles) if sim_cycles else frequency_hz * sim_seconds
cycles = cycles.to_integral_value(rounding=ROUND_HALF_UP)
if period <= 0 or cycles <= 0:
    raise SystemExit("clock period and simulation cycles must be positive")
print(int(period), int(cycles))
PY
)

sed -E -e "s/^clock_period_ps:.*/clock_period_ps: ${CLOCK_PERIOD_PS}/" \
       -e "s/^simulation_time:.*/simulation_time: ${SIM_CYCLES}/" \
       "${CONFIG}" > "${RUN_CONFIG}"

echo "[1/3] Generating traffic table (${FREQUENCY_HZ} Hz, ${SIM_CYCLES} cycles)"
"${HERE}/snn_to_noxim_traffic.py" \
  --snn "${SNN}" \
  --partition "${PARTITION}" \
  --placement "${PLACEMENT}" \
  --output "${TRAFFIC}" \
  --mesh-dim-x "${MESH_DIM_X}" \
  --mesh-dim-y "${MESH_DIM_Y}" \
  --clock-period-ps "${CLOCK_PERIOD_PS}" \
  --fanout-mode "${FANOUT_MODE}" \
  --meta-output "${TRAFFIC_META}"

echo "[2/3] Running Noxim (${MESH_DIM_X}x${MESH_DIM_Y} mesh, ${ROUTING} routing)"
(
  cd "${NOXIM_DIR}"
  NOXIM_PROGRESS_INTERVAL_CYCLES="${PROGRESS_INTERVAL}" \
  LD_LIBRARY_PATH="${NOXIM_DIR}/bin/libs/systemc-2.3.1/lib-linux64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${NOXIM_DIR}/bin/noxim" \
      -config "${RUN_CONFIG}" \
      -power "${POWER}" \
      -traffic table "${TRAFFIC}" \
      -dimx "${MESH_DIM_X}" \
      -dimy "${MESH_DIM_Y}" \
      -routing "${ROUTING}" \
      -sim "${SIM_CYCLES}" \
      -warmup "${WARMUP_CYCLES}" \
      -seed "${SEED}" \
      -stats_file "${RESULT_JSON}" \
    | tee "${RESULT_TXT}"
)

echo "[3/3] Summarizing"
"${HERE}/summarize_noxim_run.py" \
  --stats "${RESULT_JSON}" \
  --traffic-meta "${TRAFFIC_META}" \
  --clock-period-ps "${CLOCK_PERIOD_PS}" \
  --warmup "${WARMUP_CYCLES}" \
  --routing "${ROUTING}" \
  --output "${SUMMARY_JSON}" \
  | tee -a "${RESULT_TXT}"

echo
echo "Text results: ${RESULT_TXT}"
echo "JSON results: ${RESULT_JSON}"
echo "Summary: ${SUMMARY_JSON}"
