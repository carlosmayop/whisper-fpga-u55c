# Stream-architecture floorplan: data-movers (HBM) → SLR0, cómputo → SLR1/SLR2.
#
# Tras el rediseño, TODO acceso a HBM está en funciones-mover (read/write_hbm,
# weight_loader_all_tiles, qkv_tile_loader, ln_load/store, write_qkv/concat).
# El cómputo (gemm, layernorm, transpose, gelu, flash attention) solo ve streams.
# Anclamos por wildcard del nombre de función (engancha las instancias anidadas de
# las 4 capas). Los streams cruzan la frontera de SLR como FIFOs estrechos (Laguna),
# en vez de buses AXI anchos → menos congestión de interconexión (hipótesis a validar).
#
# Regiones (plataforma xcu55c, región dinámica):
#   SLR0: CLOCKREGION_X0Y0:X6Y0 + X0Y1:X3Y1 + X5Y1:X5Y1 + X0Y2:X5Y3
#   SLR1: CLOCKREGION_X0Y4:X3Y4 + X5Y4:X5Y4 + X0Y5:X5Y7
#   SLR2: CLOCKREGION_X0Y8:X6Y10 + X0Y11:X7Y11

# --- SLR0: TODOS los data-movers (dueños de los puertos m_axi de HBM) ---
create_pblock pblock_movers_slr0
resize_pblock pblock_movers_slr0 -add {CLOCKREGION_X0Y0:CLOCKREGION_X6Y0 CLOCKREGION_X0Y1:CLOCKREGION_X3Y1 CLOCKREGION_X5Y1:CLOCKREGION_X5Y1 CLOCKREGION_X0Y2:CLOCKREGION_X5Y3}
set movers [get_cells -hier -filter {NAME =~ *read_hbm_to_stream* || NAME =~ *write_stream_to_hbm* || NAME =~ *weight_loader_all_tiles* || NAME =~ *qkv_tile_loader* || NAME =~ *ln_load* || NAME =~ *ln_store* || NAME =~ *write_qkv_with_bias* || NAME =~ *write_concat_to_hbm*}]
if {[llength $movers] == 0} { error "floorplan_streamarch: no movers matched" }
puts "floorplan_streamarch: pinning [llength $movers] mover cell(s) to SLR0"
add_cells_to_pblock pblock_movers_slr0 $movers

# --- SLR1: cómputo de atención (flash) ---
create_pblock pblock_attn_slr1
resize_pblock pblock_attn_slr1 -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y4 CLOCKREGION_X5Y4:CLOCKREGION_X5Y4 CLOCKREGION_X0Y5:CLOCKREGION_X5Y7}
set attn [get_cells -hier -filter {NAME =~ *flash_attention_compute*}]
if {[llength $attn] == 0} { error "floorplan_streamarch: no attention compute matched" }
puts "floorplan_streamarch: pinning [llength $attn] attention-compute cell(s) to SLR1"
add_cells_to_pblock pblock_attn_slr1 $attn

# --- SLR2: resto del cómputo (gemm, layernorm, transpose, gelu) ---
create_pblock pblock_compute_slr2
resize_pblock pblock_compute_slr2 -add {CLOCKREGION_X0Y8:CLOCKREGION_X6Y10 CLOCKREGION_X0Y11:CLOCKREGION_X7Y11}
set comp [get_cells -hier -filter {NAME =~ *gemm_compute_all_tiles* || NAME =~ *ln_compute* || NAME =~ *transpose_residual_compute* || NAME =~ *transpose_gelu_compute*}]
if {[llength $comp] == 0} { error "floorplan_streamarch: no SLR2 compute matched" }
puts "floorplan_streamarch: pinning [llength $comp] compute cell(s) to SLR2"
add_cells_to_pblock pblock_compute_slr2 $comp

# Sin CONTAIN_ROUTING: los streams cruzan libremente; solo fijamos dónde va la lógica.
