# Stream-arch floorplan + small_weights anclado a SLR0.
#
# streamarch_smallw con el floorplan stream-arch normal dio 149K overlaps: el
# desempaquetado de small_weights (lee el buffer HBM y rellena los 38 arrays
# locales l*_ln*/l*_*_bias) NO estaba floorplaneado → flotaba por SLR1/SLR2 y
# congestionaba. Aquí lo anclamos a SLR0 junto a los demás data-movers (es
# lectura de HBM), para que no contamine el cómputo de SLR1/SLR2.
#
# Regiones: SLR0 = CLOCKREGION_X0Y0:X6Y0 + X0Y1:X3Y1 + X5Y1 + X0Y2:X5Y3
#           SLR1 = X0Y4:X3Y4 + X5Y4 + X0Y5:X5Y7
#           SLR2 = X0Y8:X6Y10 + X0Y11:X7Y11

# --- SLR0: data-movers + desempaquetado/arrays de small_weights ---
create_pblock pblock_movers_slr0
resize_pblock pblock_movers_slr0 -add {CLOCKREGION_X0Y0:CLOCKREGION_X6Y0 CLOCKREGION_X0Y1:CLOCKREGION_X3Y1 CLOCKREGION_X5Y1:CLOCKREGION_X5Y1 CLOCKREGION_X0Y2:CLOCKREGION_X5Y3}
set movers [get_cells -hier -filter {NAME =~ *read_hbm_to_stream* || NAME =~ *write_stream_to_hbm* || NAME =~ *weight_loader_all_tiles* || NAME =~ *qkv_tile_loader* || NAME =~ *ln_load* || NAME =~ *ln_store* || NAME =~ *write_qkv_x3* || NAME =~ *write_concat_to_hbm*}]
puts "floorplan_sw0: movers matched = [llength $movers]"
add_cells_to_pblock pblock_movers_slr0 $movers

# small_weights: el buffer HBM y los arrays locales desempaquetados (l0..l3 ln/bias)
set sw [get_cells -hier -filter {NAME =~ *small_weights* || NAME =~ *_ln1_g* || NAME =~ *_ln1_b* || NAME =~ *_ln2_g* || NAME =~ *_ln2_b* || NAME =~ *_q_bias* || NAME =~ *_v_bias* || NAME =~ *_attn_out_bias* || NAME =~ *_ffn1_bias* || NAME =~ *_ffn2_bias*}]
puts "floorplan_sw0: small_weights cells matched = [llength $sw]"
if {[llength $sw] > 0} { add_cells_to_pblock pblock_movers_slr0 $sw }

# --- SLR1: cómputo de atención ---
create_pblock pblock_attn_slr1
resize_pblock pblock_attn_slr1 -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y4 CLOCKREGION_X5Y4:CLOCKREGION_X5Y4 CLOCKREGION_X0Y5:CLOCKREGION_X5Y7}
set attn [get_cells -hier -filter {NAME =~ *flash_attention_compute*}]
puts "floorplan_sw0: attention cells matched = [llength $attn]"
add_cells_to_pblock pblock_attn_slr1 $attn

# --- SLR2: resto del cómputo (gemm, layernorm, transpose, gelu) ---
create_pblock pblock_compute_slr2
resize_pblock pblock_compute_slr2 -add {CLOCKREGION_X0Y8:CLOCKREGION_X6Y10 CLOCKREGION_X0Y11:CLOCKREGION_X7Y11}
set comp [get_cells -hier -filter {NAME =~ *gemm_compute_all_tiles* || NAME =~ *ln_compute* || NAME =~ *transpose_residual_compute* || NAME =~ *transpose_gelu_compute*}]
puts "floorplan_sw0: SLR2 compute cells matched = [llength $comp]"
add_cells_to_pblock pblock_compute_slr2 $comp
