
echo "Running script 1..."
python3 1_0_main_original.py

echo "Running script2..."
python3 2_0_main_original_met_tiling.py

echo "Running script3..."
python3 2_1_first_port_to_open_cl_tiling_but_no_local_memory_constant_tile_size.py

echo "Running script 4.."
python3 2_2_no_local_memory_constant_workgroup_size_variable_tile_size.py

echo "Running script 5..."
python3 2_3_no_local_memory_varying_workgroup_size_constant_tile_size.py