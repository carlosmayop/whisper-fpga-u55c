# Rediseño "stream architecture" — data-movers en SLR0, cómputo en SLR1/2

Objetivo (idea de Carlos): que **todo el acceso a HBM viva en SLR0** (donde está
físicamente la HBM) y que el **cómputo viva en SLR1/SLR2**, conectados por
`hls::stream`. Así el **cruce entre SLR es un FIFO estrecho** (handshake
data+valid/ready, que Vivado pipelinea con registros Laguna) en lugar de un
**bus AXI ancho** (direcciones + datos + motor de lectura), que es lo que
congestiona la interconexión.

## Punto de partida (clave): el kernel YA es stream-based por fase
Cada fase del encoder es una región `#pragma HLS DATAFLOW` con la forma:

    read_hbm_to_stream(buffer_HBM, stream) → [cómputo sobre stream] → write_stream_to_hbm(stream, buffer_HBM)

Fases: phase_qkv_projection, process_head_dual (atención FA), phase_wo_and_residual,
phase_ffn_fc1_gelu, phase_ffn_fc2_residual. Los movers (`read_hbm_to_stream` /
`write_stream_to_hbm`) y los adaptadores `m_axi` (`gmem_*_m_axi_U`) ya son
instancias discretas. → El trasplante a "movers en SLR0 / cómputo en SLR1-2"
es sobre todo **floorplanning + ajuste de FIFOs de cruce**, NO una reescritura total.

## Plan incremental (validando en cada paso)
1. **Exp A (sin reescribir código):** floorplan que ancla TODOS los adaptadores
   `m_axi`/movers HBM a SLR0 y el cómputo (fases + atención) a SLR1/SLR2, sobre el
   xo unroll-16 ya validado. Mide si los cruces solo-stream mejoran SLL/timing/routabilidad
   frente al floorplan de bloques D3 (que cruza buses AXI completos y aun así ruteó).
2. **Exp B (ajuste de código ligero):** subir la profundidad de los `hls::stream` que
   cruzan SLR (FIFOs más profundos → toleran la latencia del cruce Laguna sin frenar el
   pipeline) y marcar esos streams. Re-csynth + link.
3. **Exp C (reestructura "pesada" real, si A/B no bastan):** consolidar los movers en una
   región data-mover dedicada que posea todos los puertos HBM y exponga un haz de streams
   a un "compute cluster" en SLR1/2. Reto: el encoder es ITERATIVO (4 capas, buffers
   ping-pong, residuales) y reutiliza el HW de cómputo → el data-mover necesita un patrón
   request/response por fase, no un flujo feed-forward único. Es el paso más invasivo.

## Por qué puede rescatar dev/v2
dev y v2 (atención original pesada) fallan con **congestión nivel 7** — su cuello es la
densidad + cruces de bus AXI. Si el cruce pasa a ser solo-stream, podrían volverse
rutables donde el floorplan de bloques falló. Es el mayor incentivo del rediseño.

## Estado
- Copia del proyecto FA (unroll-16) creada. Exp A en preparación.

## Datos csynth (stream-arch + atención reestructurada)
- Fmax 342.47 MHz, periodo 2.92 ns (target 4.0). Sin pérdida vs FA base.
- Recursos: 477.803 LUT (109% 1 SLR, 36% device), 1290 DSP, 542 BRAM, 16 URAM.
  +~78K LUT vs FA base C1 (400K) = coste de los FIFOs de cruce.
- Latencia por instancia (= idéntica a FA base, el split no penaliza):
  layer_norm 2.59M ×2/capa | qkv 9.38M | atención(process_head_dual) 24.45M ×3/capa
  | wo 3.92M | ffn1/ffn2 "?" (~12-13M est). Top y FFN salen "?" (dataflow no acotable).
- Estimación: ~117M ciclos/capa → encoder 4 capas ~468M → ~1.37s@342MHz / ~1.87s@250MHz.
- Atención domina (~63%), por el unroll-16/II=4 (metido para routabilidad).

## PENDIENTE (valorar cuando termine el link streamarch)
Si el stream-crossing rutea con holgura, probar volver a UNROLL factor=32 / II=2 en
S_phase y acc_phase de multi_head_att.cpp → atención ~12M/llamada (≈2× más rápido,
encoder ~0.9s) a costa de ~+75K LUT. El cruce solo-stream podría absorber esa densidad
donde el floorplan de bloques (C2) obligó a bajar a unroll-16.
