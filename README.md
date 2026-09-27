# Aceleración del encoder de Whisper en FPGA (Alveo U55C)

Implementación en HLS del encoder de [Whisper](https://github.com/openai/whisper) sobre una
AMD Alveo U55C, integrada en [whisper.cpp](https://github.com/ggml-org/whisper.cpp), con el
arnés de medida de energía usado para comparar CPU, GPU y FPGA.

El repositorio contiene **dos aceleradores independientes**:

| | Qué hace | Modelo | Kernel | Reloj |
|---|---|---|---|---|
| [`encoder/`](encoder/) | Encoder completo (conv + capas transformer) | **`tiny`** (`D_MODEL=384`) | `whisper_encoder_top` | 127,9 MHz |
| [`attention/`](attention/) | Solo la atención multi-cabeza, Q·K en punto fijo | **`small`** (`QKV_CHANNELS=2304`) | `attention_top` | 267,2 MHz |

> **Cada acelerador está dimensionado para un modelo distinto y no son intercambiables.** Los
> parámetros están fijados en tiempo de compilación en `src/whisper_types.h`. Ejecutar el encoder
> con `small.en` no da error: carga, ejecuta y devuelve basura, y el decoder no emite texto.

El de atención es el que se integra de forma transparente en whisper.cpp: sustituye únicamente el
bloque de atención del encoder y deja el resto del pipeline en CPU.

## Requisitos

- **AMD Vitis / Vitis HLS 2025.2** (otras versiones no están probadas)
- **XRT 2.19** y la plataforma `xilinx_u55c_gen3x16_xdma_3_202210_1`
- **Alveo U55C**. Sintetizar para esta parte requiere licencia de Vitis; las partes Alveo están
  cubiertas por la licencia gratuita, pero otras familias necesitan Enterprise.
- **Los modelos que toque**: `ggml-small.en.bin` para el acelerador de atención y
  `ggml-tiny-q5_1.bin` para el encoder completo (ver la tabla de arriba).
- Para el banco de CPU con ggml: un árbol de whisper.cpp ya compilado (se enlaza contra sus
  `libggml-base` y `libggml-cpu`).

## Estructura

```
encoder/            acelerador del encoder completo
  src/              fuentes HLS y testbenches
  host/             biblioteca de host XRT + stub para pruebas en CPU
  tools/            extracción de pesos y comparación de capas (Python)
  hls_config.cfg    configuración de síntesis (v++ -c)
  conn.cfg          conectividad HBM y floorplan (v++ -l)
attention/          acelerador de atención, misma estructura
  attn_bench_small.cpp   banco de medida sobre la tarjeta
whisper.cpp/        parches contra el upstream
bench/              arnés de medida de energía y bancos de CPU
```

Los dos aceleradores comparten casi todo el código: `host/` es idéntico y de `src/` solo difieren
`encoder_top.cpp`, `multi_head_att.cpp` y `whisper_types.h`. Se mantienen como árboles separados
porque cada uno se sintetiza con un *top* distinto (`whisper_encoder_top` frente a `attention_top`)
y con parámetros de tamaño distintos, y así cada directorio es autocontenido.

## 1. Construir un acelerador

El flujo es el mismo para los dos; cambia el directorio y el nombre del kernel.

```bash
source /ruta/a/Vitis/2025.2/settings64.sh
source /opt/xilinx/xrt/setup.sh

cd attention
# Paso 1: C++ -> objeto de kernel (.xo)
v++ -c --mode hls --config hls_config.cfg --work_dir _xo_work
cp _xo_work/attention_top.xo .

# Paso 2: enlazado -> bitstream (.xclbin). Tarda varias horas.
v++ -l -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 \
    --config conn.cfg --freqhz 300000000:attention_top_1 --save-temps \
    -o attention.xclbin attention_top.xo
```

Para el encoder completo, desde `encoder/`, con `--freqhz 250000000:whisper_encoder_top_1` y
`-o whisper_encoder.xclbin`.

`conn.cfg` fija el reparto de canales HBM (`[connectivity] sp=`) y las directivas de Vivado.
Repartir los bancos HBM en canales alternos fue lo que permitió rutar el diseño.

> **El `conn.cfg` del encoder lleva un marcador que hay que sustituir.** Referencia el floorplan
> por SLR con una ruta absoluta, así que antes de enlazar:
> ```bash
> sed -i "s|@REPO@|$(cd .. && pwd)|" encoder/conn.cfg
> ```
> El de `attention/` no lo necesita: no usa floorplan manual.

## 2. Parchear y construir whisper.cpp

Los parches se aplican sobre el commit **`aa1bc0d1`** del upstream. Son dos configuraciones
alternativas, no acumulativas: aplica **una de las dos** sobre un árbol limpio.

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
git checkout aa1bc0d1
git apply /ruta/a/whisper-fpga-u55c/whisper.cpp/0002-attention-fpga.patch
```

| Parche | Qué añade |
|---|---|
| `0001-encoder-fpga.patch` | Descarga del **encoder completo** (`fpga_backend`, opción `WHISPER_USE_FPGA`) |
| `0002-attention-fpga.patch` | Lo anterior **más** la descarga de **solo la atención** (`whisper_fpga_attn`) |

Antes de compilar whisper.cpp hay que construir la biblioteca de host, que es la que habla con XRT:

```bash
cd /ruta/a/whisper-fpga-u55c/encoder/host
/usr/bin/cmake -B build_xrt && /usr/bin/make -C build_xrt -j$(nproc)
# variante sin tarjeta, que ejecuta el mismo código en CPU:
/usr/bin/cmake -B build_stub -DFPGA_STUB=ON && /usr/bin/make -C build_stub -j$(nproc)
```

Y después whisper.cpp apuntando a ella:

```bash
cd /ruta/a/whisper.cpp
/usr/bin/cmake -B build-fpga-attn -DWHISPER_USE_FPGA=ON \
    -DFPGA_ENCODER_INCLUDE_DIR=/ruta/a/whisper-fpga-u55c/encoder/host \
    -DFPGA_ENCODER_LIB_DIR=/ruta/a/whisper-fpga-u55c/encoder/host/build_xrt
/usr/bin/cmake --build build-fpga-attn -j$(nproc) --target whisper-cli
```

> **Usa `/usr/bin/cmake` explícitamente.** Tras `source settings64.sh`, `cmake` resuelve al de
> Vivado (3.24.2), que en Ubuntu 24.04 muere con `libssl.so.10: cannot open shared object file`.

## 3. Ejecutar

La aceleración es **opt-in por variable de entorno**. Sin ella, el binario es whisper.cpp normal.

```bash
VITIS_LNX64=/ruta/a/Vitis/2025.2/lnx64
export LD_LIBRARY_PATH=$VITIS_LNX64/tools/fpo_v7_1:$VITIS_LNX64/lib/csim:$LD_LIBRARY_PATH
```

**Atención en FPGA** (el resto del pipeline sigue en CPU). Va por su propia ruta XRT, así que no
depende de la biblioteca del encoder:

```bash
WHISPER_FPGA_ATTN_XCLBIN=/ruta/a/attention.xclbin \
  ./build-fpga-attn/bin/whisper-cli -m models/ggml-small.en.bin -f samples/jfk.wav -l en
```
```
[fpga-attn] xclbin cargado: ...
[fpga-attn] buffers reservados (n_state=768, 6.9 MB/banco qkv)
[fpga-attn PERFIL] 12 llamadas (capas), 24 pasadas de kernel
  kernel HW puro :    958.5 ms  (39.94 ms/pasada)
```

**Encoder completo en FPGA.** Necesita los pesos extraídos con `tools/extract_weights.py` y, sobre
todo, **enlazar contra `build_xrt` y no contra `build_stub`**: si whisper.cpp se configuró apuntando
al stub, el código se ejecuta en CPU aunque le pases el xclbin. Se puede corregir sin recompilar,
porque la dependencia se resuelve por `RUNPATH` y `LD_LIBRARY_PATH` tiene prioridad:

```bash
export LD_LIBRARY_PATH=/ruta/a/encoder/host/build_xrt:$LD_LIBRARY_PATH
WHISPER_FPGA_XCLBIN=/ruta/a/whisper_encoder.xclbin \
WHISPER_FPGA_WEIGHTS=/ruta/a/test_vectors_real \
  ./build-fpga-attn/bin/whisper-cli -m models/ggml-tiny-q5_1.bin -f samples/jfk.wav -l en
```
```
[whisper_fpga] Kernel loaded from ...
[whisper_fpga] Ready.
[whisper_fpga] kernel run.start->run.wait = 3011.65 ms
```

Comprueba con `ldd` cuál de las dos bibliotecas se está cargando:
```bash
ldd ./build-fpga-attn/bin/whisper-cli | grep fpga_encoder
```

Las dos bibliotecas se distinguen en la salida: el stub imprime el prefijo `[whisper_fpga_stub]` y
no enlaza XRT; la de hardware imprime `[whisper_fpga]` y sí. Si ves `Allocating HBM buffers...`
estás en la tarjeta.

La descarga del encoder completo es **opt-in**: solo se intenta si defines `WHISPER_FPGA_XCLBIN` o
`WHISPER_FPGA_WEIGHTS`. Así, usando solo el acelerador de atención no se intenta inicializar el
encoder ni aparecen avisos de un fallo que no importa.

> **Comprueba siempre que no ha caído a CPU.** Si el host no puede abrir la tarjeta, whisper.cpp
> imprime `[fpga-attn] init FALLÓ ... usando CPU` y **sigue funcionando en CPU sin más aviso**. Es
> muy fácil creer que estás midiendo la FPGA cuando no lo estás.

El banco sobre la tarjeta, sin whisper.cpp de por medio:

```bash
cd attention
./attn_bench_small <xclbin> [iters] [head_base] [patron]
#   patron: const (todo a 1.0) | rand (aleatorio, conmutación realista)
```

## 4. Medir energía

```bash
cd bench
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj   # se pierde al reiniciar

./measure_energy.sh -b 20 -- <comando>          # una ejecución

# El barrido necesita saber dónde está cada cosa (argumentos: repeticiones, delay, baseline)
export WCUDA=/ruta/a/whisper.cpp              # árbol con CUDA
export WFPGA=/ruta/a/whisper.cpp-fpga-attn    # árbol con el parche de atención
export XCLBIN=/ruta/a/attention.xclbin
export ATTN_BENCH=/ruta/a/attn_bench_small
EXPS=attn ./sweep_energia.sh 1 300 20
```

El banco de CPU con ggml se compila contra el whisper.cpp que ya tengas:

```bash
g++ -O3 -march=native -std=c++17 ggml_attn_bench.cpp -o ggml_attn_bench \
    -I$WCUDA/ggml/include -L$WCUDA/build/ggml/src -lggml-base -lggml-cpu \
    -Wl,-rpath,$WCUDA/build/ggml/src -lm
CHECK=1 ./ggml_attn_bench 20 8     # CHECK verifica el grafo contra una referencia propia
```

`measure_energy.sh` mide una línea base en reposo, ejecuta el comando muestreando las tres
plataformas e integra por trapecios. `sweep_energia.sh` compara configuraciones emparejadas
(mismo binario, cambiando una sola variable) y avisa si las líneas base se han desplazado entre
medidas.

## Licencia

MIT, ver [LICENSE](LICENSE). Los parches de `whisper.cpp/` derivan de whisper.cpp, que también es
MIT, así que no hay fricción entre ambos.
