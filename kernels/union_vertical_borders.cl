// (Ik heb pathcompression weggelaten)
inline int find_root(__global int* parent, int i) {
    if (i == -1) {
        return -1;
    }
    int root = i;
    while (parent[root] != root && parent[root] != -1) {
        root = parent[root];
    }
    return root;
}

__kernel void union_vertical_borders(__global const int* mask,
                                     __global int* parent,
                                     const int width,
                                     const int height,
                                     const int tile_size,
                                     __global int* changes_made) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // We voeren deze kernel alleen uit voor pixels op de laatste van een tile
    // (behalve voor de allerlaatste kolom van de hele afbeelding want deze heeft geen rand met een andere tile).
    if ((x % tile_size != tile_size - 1) || (x == width - 1)) {
        return;
    }

    int idx_a = y * width + x;
    if (mask[idx_a] == -1) return;

    int root_a = find_root(parent, idx_a);
    if (root_a == -1) return;

    // Controleer de 3 buren in de kolom ernaast (x+1)
    for (int dy = -1; dy <= 1; dy++) {
        int nx = x + 1;
        int ny = y + dy;

        // Controleer of de buur binnen de afbeelding valt
        if (ny >= 0 && ny < height) {
            int idx_b = ny * width + nx;
            
            if (mask[idx_b] != -1) {
                int root_b = find_root(parent, idx_b);

                if (root_b != -1 && root_a != root_b) {
                    // We doen een veilige, atomaire union-operatie
                    int old_val;
                    int new_root;
                    if (root_a < root_b) {
                        new_root = root_a;
                        old_val = atomic_min(&parent[root_b], new_root);
                    } else {
                        new_root = root_b;
                        old_val = atomic_min(&parent[root_a], new_root);
                    }
                    if (old_val > new_root) {
                        atomic_or(changes_made, 1);
                    }
                }
            }
        }
    }
}