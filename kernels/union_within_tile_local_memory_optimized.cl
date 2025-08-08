// De definitieve, werkende versie met __local geheugen en "slimme kopie".

inline int find_root_local(__local int* parent, int i) {
    if (i == -1) return -1;
    int root = i;
    while (parent[root] != root && parent[root] != -1) {
        root = parent[root];
    }
    return root;
}

__kernel void union_within_tile_optimized(__global const int* mask_global,
                                __global int* parent_global,
                                const int width,
                                const int height,
                                const int tile_size,
                                __global int* changes_made,
                                __local int* parent_local) {

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx = get_group_id(0) * tile_size + lx;
    int gy = get_group_id(1) * tile_size + ly;
    int local_idx = ly * tile_size + lx;

    // Basiscoördinaten van de linkerbovenhoek van de huidige tegel.
    int group_gx_base = get_group_id(0) * tile_size;
    int group_gy_base = get_group_id(1) * tile_size;

    // STAP 1: "SLIMME KOPIE" MET INDEX-VERTALING
    if (gx < width && gy < height) {
        int global_idx = gy * width + gx;
        if (mask_global[global_idx] == -1) {
            parent_local[local_idx] = -1;
        } else {
            int global_parent_idx = parent_global[global_idx];
            int parent_gx = global_parent_idx % width;
            int parent_gy = global_parent_idx / width;

            // Check of de parent BINNEN de huidige tegel valt.
            if (parent_gx >= group_gx_base && parent_gx < group_gx_base + tile_size &&
                parent_gy >= group_gy_base && parent_gy < group_gy_base + tile_size) {
                // JA: Vertaal de globale parent index naar een LOKALE index.
                int parent_lx = parent_gx - group_gx_base;
                int parent_ly = parent_gy - group_gy_base;
                parent_local[local_idx] = parent_ly * tile_size + parent_lx;
            } else {
                // NEE: De parent ligt buiten de tegel. Maak deze pixel zijn eigen lokale root.
                parent_local[local_idx] = local_idx;
            }
        }
    } else {
        parent_local[local_idx] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // STAP 2: Union-Find in lokaal geheugen (deze logica is nu veilig)
    if (parent_local[local_idx] != -1) {
        // ... (deze code is onveranderd en werkt nu correct) ...
        int root_a = find_root_local(parent_local, local_idx);
        if (root_a != -1) {
            for (int dy = -1; dy <= 1; dy++) { for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nlx = lx + dx; int nly = ly + dy;
                if (nlx >= 0 && nlx < tile_size && nly >= 0 && nly < tile_size) {
                    int neighbor_idx = nly * tile_size + nlx;
                    if (parent_local[neighbor_idx] != -1) {
                        int root_b = find_root_local(parent_local, neighbor_idx);
                        if (root_b != -1 && root_a != root_b) {
                            if (root_a < root_b) atomic_min(&parent_local[root_b], root_a);
                            else atomic_min(&parent_local[root_a], root_b);
                        }
                    }
                }
            }}
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // STAP 3: Schrijf resultaat terug (deze logica is nu ook veilig)
    if (gx < width && gy < height) {
        int global_idx = gy * width + gx;
        int final_local_root = find_root_local(parent_local, local_idx);
        if (final_local_root != -1) {
            int root_lx = final_local_root % tile_size;
            int root_ly = final_local_root / tile_size;
            int final_global_root = (group_gy_base + root_ly) * width + (group_gx_base + root_lx);
            
            // We updaten niet zomaar de parent, we doen een 'union' op de globale structuur.
            // Dit is essentieel om de info van de lokale pass correct te mergen.
            int global_root_of_pixel = global_idx;
            while(parent_global[global_root_of_pixel] != global_root_of_pixel) {
                global_root_of_pixel = parent_global[global_root_of_pixel];
            }

            if (final_global_root < global_root_of_pixel) {
                int old_val = atomic_min(&parent_global[global_root_of_pixel], final_global_root);
                if(old_val > final_global_root) atomic_or(changes_made, 1);
            } else if (global_root_of_pixel < final_global_root) {
                int old_val = atomic_min(&parent_global[final_global_root], global_root_of_pixel);
                 if(old_val > global_root_of_pixel) atomic_or(changes_made, 1);
            }
        }
    }
}