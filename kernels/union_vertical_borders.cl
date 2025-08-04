__kernel void union_vertical_borders(__global int* mask,
                                     __global int* parent,
                                     const int width,
                                     const int height,
                                     const int tile_size) {
    int col = get_global_id(0);

    if (col >= width)
        return;

    for (int tile = 1; tile < height / tile_size; tile++) {
        int row_a = tile * tile_size - 1;
        int row_b = tile * tile_size;

        int idx_a = row_a * width + col;
        int idx_b = row_b * width + col;

        int a = mask[idx_a];
        int b = mask[idx_b];

        if (a == -1 || b == -1)
            continue;

        // --- find root of a ---
        int root_a = a;
        while (parent[root_a] != root_a)
            root_a = parent[root_a];

        // --- find root of b ---
        int root_b = b;
        while (parent[root_b] != root_b)
            root_b = parent[root_b];

        if (root_a != root_b) {
            // Maak root_b kind van root_a (atomic!)
            atomic_min(&parent[root_b], root_a);
        }
    }
}
