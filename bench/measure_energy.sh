#!/bin/bash
# Measures the energy consumed by CPU (RAPL), GPU (nvidia-smi) and FPGA (xrt-smi) while
# a command runs, and subtracts the idle baseline.
#
#   ./measure_energy.sh [-b BASELINE_SECS] [-o OUTPUT.csv] -- <command...>
#
# CPU: RAPL package-0 integrating counter (cores+uncore, does NOT include DRAM). Needs permission:
#   sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj     (lost on reboot)
# GPU/FPGA: instantaneous power sampled and integrated by trapezoids. The FPGA reports
# CARD power (includes HBM and regulators); the GPU reports board power.
set -u
set -m

BASELINE_S=5
OUT_CSV=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b) BASELINE_S="$2"; shift 2 ;;
        -o) OUT_CSV="$2"; shift 2 ;;
        --) shift; break ;;
        *)  echo "uso: $0 [-b segs] [-o csv] -- <comando...>" >&2; exit 1 ;;
    esac
done
[[ $# -eq 0 ]] && { echo "falta el comando tras --" >&2; exit 1; }

RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
RAPL_MAX=$(cat /sys/class/powercap/intel-rapl:0/max_energy_range_uj 2>/dev/null || echo 0)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

have_gpu=0; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && have_gpu=1

# Card power: we prefer the hwmon node of the xocl driver (a sysfs read, ~0.9 ms)
# over "xrt-smi examine" (a fork+exec of ~240 ms). Measured 18-sep-2026: the
# xrt-smi sampler cost +28.4 W of CPU, more than the whole machine at idle,
# and it leaked into every baseline. With hwmon the overhead drops to +1.3 W.
FPGA_PWR=""
for d in /sys/bus/pci/drivers/xocl/*/hwmon/hwmon*/power1_input; do
    [[ -r $d ]] && { FPGA_PWR=$d; break; }
done
have_fpga=0
if [[ -n $FPGA_PWR ]]; then have_fpga=1
elif command -v xrt-smi >/dev/null; then have_fpga=2; fi   # 2 = fallback via xrt-smi
have_cpu=0; [[ -r $RAPL ]] && have_cpu=1

# A single nvidia-smi process in loop mode; the while-read uses builtins only.
sample_gpu() {
    nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 50 2>/dev/null \
      | while read -r p; do printf '%s %s\n' "$EPOCHREALTIME" "$p"; done
}
# Direct sysfs read with builtins: no date, no cat, no external sleep.
# The "read -t" on a read/write fd acts as a sleep without forking.
sample_fpga() {
    exec 9<> <(:)
    local p
    while :; do
        read -r p < "$FPGA_PWR" || break
        printf '%s %s\n' "$EPOCHREALTIME" "$p"      # microvatios
        read -t 0.05 -u 9 || true
    done
}
sample_fpga_xrtsmi() { while :; do
        printf '%s %s\n' "$(date +%s.%N)" \
            "$(xrt-smi examine --report electrical 2>/dev/null | awk '/^  Power  /{print $3; exit}')"
        sleep 0.05
    done; }

# Trapezoidal integral of (t,W) -> joules, plus mean power.
integrate() { awk -v k="${2:-1}" '
    BEGIN { n=0 }
    NF==2 && $2+0==$2 { t[n]=$1; p[n]=$2/k; n++ }
    END { if (n<2) { print "0 0 0"; exit }
          e=0; for (i=1;i<n;i++) e += (p[i]+p[i-1])/2*(t[i]-t[i-1])
          printf "%.3f %.3f %.3f\n", e, e/(t[n-1]-t[0]), t[n-1]-t[0] }' "$1"; }

start_samplers() {
    [[ $have_gpu  == 1 ]] && { sample_gpu  > "$1/gpu.dat"  2>/dev/null & echo $! > "$1/gpu.pid"; }
    [[ $have_fpga == 1 ]] && { sample_fpga        > "$1/fpga.dat" 2>/dev/null & echo $! > "$1/fpga.pid"; }
    [[ $have_fpga == 2 ]] && { sample_fpga_xrtsmi > "$1/fpga.dat" 2>/dev/null & echo $! > "$1/fpga.pid"; }
}
stop_samplers() {
    local f pid
    for f in "$1"/*.pid; do
        [[ -e $f ]] || continue
        pid=$(cat "$f")
        kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    done
    wait 2>/dev/null
}

#  idle baseline 
echo "== midiendo reposo (${BASELINE_S}s) =="
mkdir -p "$TMP/base"
start_samplers "$TMP/base"
[[ $have_cpu == 1 ]] && cpu0=$(cat $RAPL)
sleep "$BASELINE_S"
[[ $have_cpu == 1 ]] && cpu1=$(cat $RAPL)
stop_samplers "$TMP/base"

base_cpu_w=0
if [[ $have_cpu == 1 ]]; then
    d=$((cpu1 - cpu0)); (( d < 0 )) && d=$((d + RAPL_MAX))
    base_cpu_w=$(awk -v d=$d -v s=$BASELINE_S 'BEGIN{printf "%.3f", d/1e6/s}')
fi
base_gpu_w=$([[ $have_gpu  == 1 ]] && integrate "$TMP/base/gpu.dat"  | awk '{print $2}' || echo 0)
FPGA_K=1; [[ $have_fpga == 1 ]] && FPGA_K=1000000      # hwmon da microvatios
base_fpga_w=$([[ $have_fpga != 0 ]] && integrate "$TMP/base/fpga.dat" "$FPGA_K" | awk '{print $2}' || echo 0)
printf "   reposo: CPU %.1f W | GPU %.1f W | FPGA %.1f W\n" "$base_cpu_w" "$base_gpu_w" "$base_fpga_w"

#  measured run 
echo "== ejecutando: $* =="
mkdir -p "$TMP/run"
start_samplers "$TMP/run"
[[ $have_cpu == 1 ]] && cpu0=$(cat $RAPL)
t0=$(date +%s.%N)
"$@" > "$TMP/run/stdout.log" 2>&1
rc=$?
t1=$(date +%s.%N)
[[ $have_cpu == 1 ]] && cpu1=$(cat $RAPL)
stop_samplers "$TMP/run"
dur=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.3f", b-a}')

cpu_j=0
if [[ $have_cpu == 1 ]]; then
    d=$((cpu1 - cpu0)); (( d < 0 )) && d=$((d + RAPL_MAX))
    cpu_j=$(awk -v d=$d 'BEGIN{printf "%.3f", d/1e6}')
fi
read gpu_j  gpu_w  _ <<< "$([[ $have_gpu  == 1 ]] && integrate "$TMP/run/gpu.dat"  || echo '0 0 0')"
read fpga_j fpga_w _ <<< "$([[ $have_fpga != 0 ]] && integrate "$TMP/run/fpga.dat" "$FPGA_K" || echo '0 0 0')"

awk -v d="$dur" -v rc="$rc" \
    -v cj="$cpu_j"  -v bcw="$base_cpu_w" \
    -v gj="$gpu_j"  -v bgw="$base_gpu_w"  -v gw="$gpu_w" \
    -v fj="$fpga_j" -v bfw="$base_fpga_w" -v fw="$fpga_w" 'BEGIN {
    cin = cj - bcw*d; gin = gj - bgw*d; fin = fj - bfw*d
    printf "\n== resultado (rc=%d, %.3f s) ==\n", rc, d
    printf "              %10s %10s %10s\n", "total J", "incr. J", "media W"
    printf "  CPU  RAPL   %10.2f %10.2f %10.2f\n", cj, cin, cj/d
    printf "  GPU  placa  %10.2f %10.2f %10.2f\n", gj, gin, gw
    printf "  FPGA tarjeta%10.2f %10.2f %10.2f\n", fj, fin, fw
    printf "  ---------------------------------------------\n"
    printf "  TOTAL       %10.2f %10.2f\n", cj+gj+fj, cin+gin+fin
    printf "\n  incr. = energia por encima del reposo (atribuible a la carga)\n" }'

if [[ $have_cpu != 1 ]]; then
    echo "  AVISO: RAPL no legible -> CPU marcada como 0 J."
    echo "         sudo chmod a+r $RAPL"
fi

if [[ -n $OUT_CSV ]]; then
    [[ -f $OUT_CSV ]] || echo "etiqueta,dur_s,cpu_j,gpu_j,fpga_j,cpu_base_w,gpu_base_w,fpga_base_w" > "$OUT_CSV"
    echo "\"$*\",$dur,$cpu_j,$gpu_j,$fpga_j,$base_cpu_w,$base_gpu_w,$base_fpga_w" >> "$OUT_CSV"
fi

echo "== stdout del comando =="
tail -20 "$TMP/run/stdout.log"
exit $rc
