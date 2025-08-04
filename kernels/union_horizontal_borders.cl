__kernel void union_horizontal_borders(__global int* mask,
                                       __global int* parent,
                                       const int width,
                                       const int height,
                                       const int tile_size) {
    int row = get_global_id(0);
    if (row >= height) return;

    for (int col = tile_size; col < width; col += tile_size) {
        int idx_a = row * width + (col - 1);
        int idx_b = row * width + col;

        int a = mask[idx_a];
        int b = mask[idx_b];

        if (a == -1 || b == -1) continue;

        int root_a = a;
        while (parent[root_a] != root_a)
            root_a = parent[root_a];

        int root_b = b;
        while (parent[root_b] != root_b)
            root_b = parent[root_b];

        if (root_a != root_b) {
            if (root_a < root_b)
                atomic_min(&parent[root_b], root_a);
            else
                atomic_min(&parent[root_a], root_b);
        }
    }
}