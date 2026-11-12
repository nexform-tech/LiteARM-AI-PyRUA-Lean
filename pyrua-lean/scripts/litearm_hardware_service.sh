#!/usr/bin/env bash
# Start the hardware service with an explicit device environment, absolute
# station file and its own log.  It never auto-enables, resets, clears faults
# or homes the hardware.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${LITEARM_HW_CONFIG:-${ROOT}/configs/litearm_real.json}"
if [[ "${CONFIG}" != /* ]]; then
  echo "LITEARM_HW_CONFIG must be absolute: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "station config not found: ${CONFIG}" >&2
  echo "copy configs/litearm_real.example.json to configs/litearm_real.json and fill it in" >&2
  exit 2
fi
MODE="${LITEARM_HW_MODE:-}"
ENV_NAME="${LITEARM_HW_DEVICE_ENV:-litearm-hw-device}"
LOG="${LITEARM_HW_LOG:-${ROOT}/runs_hw/service.log}"

mkdir -p "$(dirname "${LOG}")"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
args=(python -m pyrualean.hardware.daemon --config "${CONFIG}" --log "${LOG}")
if [[ -n "${MODE}" ]]; then
  args+=(--execution-mode "${MODE}")
fi
echo "starting hardware service: env=${ENV_NAME} config=${CONFIG} mode=${MODE:-from-config} log=${LOG}" >&2
exec conda run --no-capture-output -n "${ENV_NAME}" "${args[@]}"
