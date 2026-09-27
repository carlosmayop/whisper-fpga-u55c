#!/bin/bash
# CPU / GPU / FPGA energy sweep over the same audio and model (Whisper small.en).
# Requires: sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
#
#   ./sweep_energia.sh [REPETITIONS] [DELAY_S] [BASELINE_S]      (default 3 60 15)
#   AUDIO=/path/audio.wav TOKENS=<n> ./sweep_energia.sh ...   to change the audio
#


set -u
N=${1:-3}
DELAY=${2:-60}
BASE=${3:-15}
# Paths: all overridable from the environment so the script stays portable.
#   HERE   directory of this script (where measure_energy.sh lives)
#   WCUDA  whisper.cpp tree built with CUDA (measures GPU and the reference CPU)
#   WFPGA  whisper.cpp tree with the FPGA attention patch
HERE=${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
WCUDA=${WCUDA:?define WCUDA=/ruta/a/whisper.cpp (build CUDA)}
WFPGA=${WFPGA:?define WFPGA=/ruta/a/whisper.cpp-fpga-attn}
MODEL=${MODEL:-$WFPGA/models/ggml-small.en.bin}
AUDIO=${AUDIO:-$WFPGA/samples/mm1.wav}      # override: AUDIO=/ruta/x.wav ./sweep_energia.sh ...
TOKENS=${TOKENS:-206}                       # tokens decodificados de ese audio (denominador tok/J)
AUDIO_S=$(python3 -c "import wave,sys; w=wave.open(sys.argv[1]); print(f'{w.getnframes()/w.getframerate():.1f}')" "$AUDIO")
XCLBIN=${XCLBIN:?define XCLBIN=/ruta/al/attention.xclbin}
ATTN_BENCH=${ATTN_BENCH:?define ATTN_BENCH=/ruta/al/attn_bench_small}
GGML_ATTN=${GGML_ATTN:-$HERE/ggml_attn_bench}             # MISMA implementacion que whisper.cpp (ggml, K/V en F16)
EXPS=${EXPS:-e2e}                           # e2e | threads | attn | all (lista con comas)
THREADS=${THREADS:-4 8 24}                  # hilos para las familias threads/attn
ATTN_ITERS=${ATTN_ITERS:-600}               # pasadas por configuracion en la familia attn
META=$HERE/energia_unidades.csv             # tag -> nº de unidades medidas, para J/unidad
CSV=$HERE/energia_resultados.csv
# libIp_floating_point_v7_1_bitacc_cmodel.so lives in lnx64/tools/fpo_v7_1 and the stub's
# RPATH does not cover it. XILINX_VITIS is exported by settings64.sh.
VITIS_LNX64=${VITIS_LNX64:-${XILINX_VITIS:+$(dirname "$XILINX_VITIS")/lnx64}}
export LD_LIBRARY_PATH=${VITIS_LNX64}/tools/fpo_v7_1:${VITIS_LNX64}/lib/csim:${LD_LIBRARY_PATH:-}

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

# The CSV is opened in append mode: if one from a previous sweep exists, move it aside.
if [[ -f $CSV ]]; then
    ts=$(date +%Y%m%d_%H%M%S)
    mv "$CSV" "${CSV%.csv}_$ts.csv"
    [[ -f $META ]] && mv "$META" "${META%.csv}_$ts.csv"
    echo "== barrido anterior apartado como ${CSV%.csv}_$ts.csv =="
fi
echo "etiqueta,unidades,unidad" > "$META"

settle() {   # reposo forzado + lectura de como va bajando la potencia
    echo "-- reposo ${DELAY}s antes de la linea base --"
    local half=$((DELAY / 2))
    sleep "$half"
    printf "   a mitad: GPU %s W | FPGA %s W\n" \
        "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)" \
        "$(xrt-smi examine --report electrical 2>/dev/null | awk '/^  Power  /{print $3; exit}')"
    sleep $((DELAY - half))
}

have_exp() { [[ $EXPS == all || ",$EXPS," == *",$1,"* ]]; }

preflight() {   # aborta si algun sensor no responde o si la fpga cae a CPU en silencio
    local fail=0
    echo "== pre-flight: comprobando que las tres plataformas son medibles =="
    if [[ -r /sys/class/powercap/intel-rapl:0/energy_uj ]]; then
        echo "   [ok]    RAPL legible"
    else
        echo "   [FALLO] RAPL no legible (se pierde en cada reinicio)"
        echo "           sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj"
        fail=1
    fi
    local w
    w=$(xrt-smi examine --report electrical 2>/dev/null | awk '/^  Power  /{print $3; exit}')
    if [[ -n $w ]]; then
        echo "   [ok]    FPGA reporta $w W"
    else
        echo "   [FALLO] xrt-smi no ve la tarjeta (permisos del nodo /dev/dri del driver xocl)"
        local node=""
        for d in /dev/dri/renderD*; do
            [[ $(basename "$(readlink -f /sys/class/drm/$(basename "$d")/device/driver)") == xocl ]] \
                && node=$d && break
        done
        echo "           sudo setfacl -m u:$USER:rw ${node:-<nodo xocl>}    # inmediato, se pierde al reiniciar"
        echo "           sudo usermod -aG render $USER                      # permanente, requiere cerrar sesion"
        fail=1
    fi
    if nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits >/dev/null 2>&1; then
        echo "   [ok]    GPU responde"
    else
        echo "   [FALLO] nvidia-smi no responde"; fail=1
    fi
    # The check that really matters: that the attention runs ON the card.
    # If the device does not open, whisper.cpp falls back to CPU silently and the fpga row measures CPU.
    # It also leaves the xclbin programmed, so the ~8 s of programming do not land
    # inside the first measured run.
    local out
    out=$(WHISPER_FPGA_ATTN_XCLBIN=$XCLBIN $WFPGA/build-fpga-attn/bin/whisper-cli \
            -m "$MODEL" -f "$WFPGA/samples/jfk.wav" -l en 2>&1)
    if grep -qi "init fall" <<< "$out"; then
        echo "   [FALLO] la configuracion fpga CAE A CPU -> la fila fpga no mediria la FPGA"
        fail=1
    else
        echo "   [ok]    la atencion se ejecuta en la tarjeta (xclbin ya programado)"
    fi
    if have_exp attn; then
        for b in "$ATTN_BENCH" "$GGML_ATTN"; do
            if [[ -x $b ]]; then echo "   [ok]    $(basename "$b") presente"
            else echo "   [FALLO] falta $b"; fail=1; fi
        done
    fi
    if (( fail )); then
        echo ""; echo "ABORTADO: arregla lo marcado [FALLO] y vuelve a lanzar."; exit 1
    fi
    echo ""
}

# UNIDADES/UNIDAD can be set before each runner; otherwise N runs is assumed.
UNIDADES=""; UNIDAD=""
WARMUP=""
fpga_w() { local d; for d in /sys/bus/pci/drivers/xocl/*/hwmon/hwmon*/power1_input; do
               [[ -r $d ]] && { awk '{printf "%.1f", $1/1e6}' "$d"; return; }; done; echo "?"; }
runner() {   # $1=etiqueta  $2..=comando repetido N veces
    local tag=$1; shift
    # the file name IS the label that measure_energy.sh writes into the CSV
    local script=/tmp/sweep_$tag.sh
    { echo '#!/bin/bash'; for i in $(seq 1 "$N"); do echo "$* >/dev/null 2>&1"; done; } > "$script"
    chmod +x "$script"
    echo ""; echo "############ $tag (x$N) ############"
    settle
    if [[ -n $WARMUP ]]; then
        # Deliberately short: leaves the xclbin loaded without warming the CPU appreciably.
        echo "-- calentando la tarjeta antes de la linea base (FPGA $(fpga_w) W) --"
        eval "$WARMUP" >/dev/null 2>&1
        sleep 5
        echo "   FPGA tras calentar: $(fpga_w) W  (si sigue <30 el run pagara la recarga)"
    fi
    WARMUP=""
    # Save the COMPLETE output per configuration: it includes the command stdout, where
    # the benchmarks report their own median. Without it you cannot tell "the kernel was
    # slower" from "there were seconds of setup" when the duration does not add up.
    "$HERE/measure_energy.sh" -b "$BASE" -o "$CSV" -- "$script" 2>&1 \
        | tee "$HERE/energia_salida_$tag.txt" \
        | sed -n '/reposo:/p;/== resultado/,/incr\. =/p'
    echo "$tag,${UNIDADES:-$N},${UNIDAD:-run}" >> "$META"
    UNIDADES=""; UNIDAD=""
    rm -f "$script"
}

preflight

#  e2e family: 4 paired configurations, end to end 
if have_exp e2e; then
    runner cpu_cuda  "$WCUDA/build/bin/whisper-cli -ng -m $MODEL -f $AUDIO -l en"
    runner gpu       "$WCUDA/build/bin/whisper-cli     -m $MODEL -f $AUDIO -l en"
    runner cpu_fpga  "env -u WHISPER_FPGA_ATTN_XCLBIN $WFPGA/build-fpga-attn/bin/whisper-cli -m $MODEL -f $AUDIO -l en"
    WARMUP="$ATTN_BENCH $XCLBIN 5 0"
    runner fpga      "env WHISPER_FPGA_ATTN_XCLBIN=$XCLBIN $WFPGA/build-fpga-attn/bin/whisper-cli -m $MODEL -f $AUDIO -l en"
fi

#  threads family: CPU scaling, end to end 
if have_exp threads; then
    for t in $THREADS; do
        runner "cpu_${t}t" "$WCUDA/build/bin/whisper-cli -ng -t $t -m $MODEL -f $AUDIO -l en"
    done
fi

#  attn family: ISOLATED attention, same unit on FPGA and CPU 
# 1 pass = 6 heads x 1504 frames in both benchmarks -> comparable J/pass.
if have_exp attn; then
    UNIDADES=$((N * ATTN_ITERS)); UNIDAD=pasada
    WARMUP="$ATTN_BENCH $XCLBIN 5 0"
    runner attn_fpga "$ATTN_BENCH $XCLBIN $ATTN_ITERS 0"
    # CPU reference: the ggml attention, which is what whisper.cpp actually runs.
    # (There is also cpu_attn_mt, an in-house implementation x1.2-1.9 slower; it was measured once
    # to quantify that gap and left out of the sweep because it is not the reference.)
    for t in $THREADS; do
        UNIDADES=$((N * ATTN_ITERS)); UNIDAD=pasada
        runner "attn_ggml_${t}t" "$GGML_ATTN $ATTN_ITERS $t"
    done
fi

#  quality control: the baselines must agree across configurations 
echo ""; echo "== lineas base por configuracion (deben ser parecidas; si no, hubo contaminacion) =="
# The FPGA baseline is NOT comparable between rows that use the card (programmed, ~36 W) and
# rows that do not (deprogrammed, ~24 W): they are two different hardware states and differ
# by ~12 W by design. The spread is checked WITHIN each group, and separately that the FPGA
# rows start with the card already warm.
awk -F, 'NR>1 { gsub(/"|\/tmp\/sweep_|\.sh/,"",$1)
                printf "  %-14s CPU %6.1f W | GPU %6.1f W | FPGA %6.1f W\n", $1,$6,$7,$8
                n++
                for(i=6;i<=7;i++){ if(n==1||$i<mn[i])mn[i]=$i; if($i>mx[i])mx[i]=$i }
                g = ($1 ~ /fpga/) ? "fpga" : "host"
                if (!(g in gn)) { gmn[g]=$8; gmx[g]=$8 }
                gn[g]++; if($8<gmn[g])gmn[g]=$8; if($8>gmx[g])gmx[g]=$8
                if (g=="fpga" && $8 < 30)
                    fria = fria sprintf("  AVISO: %s midio con la tarjeta a %.1f W (desprogramada) -> esa fila paga ~8 s de recarga del bitstream y su incremental incluye la estatica; NO comparable\n", $1, $8) }
     END { printf "\n  dispersion max-min: CPU %.1f W | GPU %.1f W", mx[6]-mn[6], mx[7]-mn[7]
           if ("fpga" in gn) printf " | FPGA(filas fpga) %.1f W", gmx["fpga"]-gmn["fpga"]
           if ("host" in gn) printf " | FPGA(resto) %.1f W", gmx["host"]-gmn["host"]
           printf "\n"
           # The baseline spread propagates into the incremental energy multiplied by the
           # duration, so a few W ruin the measurement.
           nm[6]="CPU"; nm[7]="GPU"
           for (j=6; j<=7; j++)
               if (mx[j]-mn[j] > 3)
                   printf "  AVISO: la base de %s varia %.1f W -> maquina no quieta o DELAY corto; la medida NO es fiable\n", nm[j], mx[j]-mn[j]
           for (g in gn)
               if (gn[g] > 1 && gmx[g]-gmn[g] > 3)
                   printf "  AVISO: la base de FPGA varia %.1f W entre filas %s -> la tarjeta cambio de estado a mitad del barrido\n", gmx[g]-gmn[g], g
           printf "%s", fria
           if (mx[6]==0) print "  AVISO: columna CPU a 0 -> RAPL no se leyo; la medida NO es valida"
           if (("fpga" in gn) && gmx["fpga"]==0) print "  AVISO: columna FPGA a 0 -> no se leyo el sensor de la tarjeta; la medida NO es valida" }' "$CSV"

echo ""; echo "== CSV: $CSV =="
if have_exp e2e || have_exp threads; then
    echo "   Audio: $(basename "$AUDIO") — $AUDIO_S s, $TOKENS tokens decodificados."
fi
echo "   Unidades medidas por configuracion ($META):"
awk -F, 'NR>1{printf "     %-14s %6s %s\n",$1,$2,$3}' "$META"
