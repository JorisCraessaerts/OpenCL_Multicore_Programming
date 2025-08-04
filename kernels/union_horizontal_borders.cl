__kernel void union_horizontal_borders(__global int* mask,
                                       __global int* parent,
                                       const int width,
                                       const int height,
                                       const int tile_size,
                                       __global int* changes_made) {
    int row = get_global_id(0);
    if (row >= height) return;

    for (int col = tile_size; col < width; col += tile_size) {
        int idx_a = row * width + (col - 1);
        int idx_b = row * width + col;

        int a = mask[idx_a];
        int b = mask[idx_b];

        if (a == -1 || b == -1) continue;

        // Check parent validity
        if (parent[a] == -1 || parent[b] == -1) continue;

        // Zoek root van a
        int root_a = a;
        while (parent[root_a] != root_a && parent[root_a] != -1)
            root_a = parent[root_a];

        // Zoek root van b
        int root_b = b;
        while (parent[root_b] != root_b && parent[root_b] != -1)
            root_b = parent[root_b];

        if (root_a == -1 || root_b == -1 || root_a == root_b)
            continue;

        int old_val;
        if (root_a < root_b) {
            old_val = atomic_min(&parent[root_b], root_a);
            if (old_val != root_a) atomic_or(changes_made, 1);
        } else {
            old_val = atomic_min(&parent[root_a], root_b);
            if (old_val != root_b) atomic_or(changes_made, 1);
        }
    }
}
