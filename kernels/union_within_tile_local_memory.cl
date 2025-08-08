#define TILE_SIZE 16
#define TILE_AREA (TILE_SIZE * TILE_SIZE)

// Hybride find_root: kijk eerst lokaal, anders globaal
inline int find_root(__global int* gparent,
                     __local int* lparent,
                     int gidx,
                     int tile_start,
                     int tile_end) {
    int root = gidx;
    while (true) {
        int p = (root >= tile_start && root < tile_end)
                  ? lparent[root - tile_start]
                  : gparent[root];

        if (p == root || p == -1) break;
        root = p;
    }
    return root;
}

__kernel void union_within_tile(__global const int* mask,
                                __global int* parent,
                                const int width,
                                const int height,
                                const int tile_size,
                                __global int* changes_made) {
    // Globale coördinaten
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx >= width || gy >= height) return;

    // Lokale coördinaten
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int lidx = ly * tile_size + lx;

    // Indexen
    int gidx = gy * width + gx;
    int tile_x = gx / tile_size;
    int tile_y = gy / tile_size;
    int tile_start = tile_y * tile_size * width + tile_x * tile_size;
    int tile_end = tile_start + tile_size * width; // overschatting is ok zolang buiten tile

    // Lokale caches
    __local int local_mask[TILE_AREA];
    __local int local_parent[TILE_AREA];

    // Initialisatie lokale tiles
    local_mask[lidx] = mask[gidx];
    local_parent[lidx] = parent[gidx];

    barrier(CLK_LOCAL_MEM_FENCE); // Sync alle reads

    if (local_mask[lidx] == -1) return;

    int root_a = find_root(parent, local_parent, gidx, tile_start, tile_end);
    if (root_a == -1) return;

    const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

    for (int k = 0; k < 8; k++) {
        int nx = gx + dx[k];
        int ny = gy + dy[k];

        if (nx >= 0 && ny >= 0 && nx < width && ny < height &&
            (nx / tile_size) == tile_x && (ny / tile_size) == tile_y) {
            
            int nidx = ny * width + nx;
            int nlx = nx % tile_size;
            int nly = ny % tile_size;
            int nlidx = nly * tile_size + nlx;

            if (local_mask[nlidx] == -1) continue;

            int root_b = find_root(parent, local_parent, nidx, tile_start, tile_end);

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
