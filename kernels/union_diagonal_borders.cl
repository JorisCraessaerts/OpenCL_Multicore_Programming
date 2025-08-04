__kernel void union_diagonal_borders(__global int* mask,
                                     __global int* parent,
                                     const int width,
                                     const int height,
                                     const int tile_size,
                                     __global int* changes_made) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= width || row >= height)
        return;

    // Diagonaal rechtsonder (row+1, col+1)
    if (row + 1 < height && col + 1 < width &&
        ((row + 1) / tile_size != row / tile_size) &&
        ((col + 1) / tile_size != col / tile_size)) {

        int idx_a = row * width + col;
        int idx_b = (row + 1) * width + (col + 1);

        int a = mask[idx_a];
        int b = mask[idx_b];
        if (a != -1 && b != -1 &&
            parent[a] != -1 && parent[b] != -1) {

            int root_a = a;
            while (parent[root_a] != root_a && parent[root_a] != -1)
                root_a = parent[root_a];

            int root_b = b;
            while (parent[root_b] != root_b && parent[root_b] != -1)
                root_b = parent[root_b];

            if (root_a != -1 && root_b != -1 && root_a != root_b) {
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
    }

    // Diagonaal linksonder (row+1, col-1)
    if (row + 1 < height && col > 0 &&
        ((row + 1) / tile_size != row / tile_size) &&
        ((col - 1) / tile_size != col / tile_size)) {

        int idx_a = row * width + col;
        int idx_b = (row + 1) * width + (col - 1);

        int a = mask[idx_a];
        int b = mask[idx_b];
        if (a != -1 && b != -1 &&
            parent[a] != -1 && parent[b] != -1) {

            int root_a = a;
            while (parent[root_a] != root_a && parent[root_a] != -1)
                root_a = parent[root_a];

            int root_b = b;
            while (parent[root_b] != root_b && parent[root_b] != -1)
                root_b = parent[root_b];

            if (root_a != -1 && root_b != -1 && root_a != root_b) {
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
    }
}
