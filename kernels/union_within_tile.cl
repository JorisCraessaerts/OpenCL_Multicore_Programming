// __kernel void union_within_tile(__global int* mask,
//                                  __global int* parent,
//                                  const int width,
//                                  const int height,
//                                  const int tile_size,
//                                  __global int* changes_made) {
//     int x = get_global_id(0);
//     int y = get_global_id(1);

//     if (x >= width || y >= height)
//         return;

//     int idx = y * width + x;
//     if (mask[idx] == -1)
//         return;

//     // Tile-ID
//     int tile_x = x / tile_size;
//     int tile_y = y / tile_size;

//     // Alle 8 buren
//     const int dx[8] = {-1, -1, -1,  0, 0, 1, 1, 1};
//     const int dy[8] = {-1,  0,  1, -1, 1, -1, 0, 1};

//     for (int k = 0; k < 8; k++) {
//         int nx = x + dx[k];
//         int ny = y + dy[k];

//         // Buiten beeld?
//         if (nx < 0 || ny < 0 || nx >= width || ny >= height)
//             continue;

//         // Buiten tile?
//         if ((nx / tile_size) != tile_x || (ny / tile_size) != tile_y)
//             continue;

//         int nidx = ny * width + nx;

//         if (mask[nidx] == -1)
//             continue;

//         // Lees ouders
//         int pa = parent[idx];
//         int pb = parent[nidx];

//         if (pa == -1 || pb == -1 || pa == pb)
//             continue;


//         int new_root = min(pa, pb);

//         int old_pa = atomic_min(&parent[pa], new_root);
//         int old_pb = atomic_min(&parent[pb], new_root);
//         int old_idx = atomic_min(&parent[idx], new_root);
//         int old_nidx = atomic_min(&parent[nidx], new_root);

//         if (old_pa != new_root || old_pb != new_root ||
//             old_idx != new_root || old_nidx != new_root) {
//             atomic_or(changes_made, 1);
//         }
//     }
// }


// Helper-functie om de root te vinden met padcompressie
// Vereenvoudigde, veilige find_root ZONDER path compression
inline int find_root(__global int* parent, int i) {
    if (i == -1) {
        return -1;
    }
    int root = i;
    // Blijf de parent volgen tot we de root vinden
    while (parent[root] != root && parent[root] != -1) {
        root = parent[root];
    }
    return root;
}

__kernel void union_within_tile(__global const int* mask,
                                __global int* parent,
                                const int width,
                                const int height,
                                const int tile_size,
                                __global int* changes_made) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    if (mask[idx] == -1) return;

    int tile_x = x / tile_size;
    int tile_y = y / tile_size;

    int root_a = find_root(parent, idx);
    if (root_a == -1) return;

    const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

    for (int k = 0; k < 8; k++) {
        int nx = x + dx[k];
        int ny = y + dy[k];

        if (nx >= 0 && ny >= 0 && nx < width && ny < height &&
            (nx / tile_size) == tile_x && (ny / tile_size) == tile_y) {
            
            int nidx = ny * width + nx;
            if (mask[nidx] != -1) {
                int root_b = find_root(parent, nidx);

                if (root_b != -1 && root_a != root_b) {
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