
echo "Running script 1..."
python3 2_1_0_main_optimized_tilling_but_without_local_memory_constant_tile_size_benchmark.py

echo "Running script2..."
python3 2_1_main_optimized_tilling_but_without_local_memory.py

echo "Running script3..."
python3 2_2_main_optimized_tiling_without_local_memory_but_varying_workgroup_sizes.py
