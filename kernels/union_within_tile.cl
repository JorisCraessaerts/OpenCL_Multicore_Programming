__kernel void union_within_tile(__global int* mask,
                                 __global int* parent,
                                 const int width,
                                 const int height,
                                 const int tile_size) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    if (mask[idx] == -1)
        return;

    // Tile-ID
    int tile_x = x / tile_size;
    int tile_y = y / tile_size;

    // Alleen buren binnen tile
    const int dx[4] = {-1, -1, -1,  0};
    const int dy[4] = {-1,  0,  1, -1};

    for (int k = 0; k < 4; k++) {
        int nx = x + dx[k];
        int ny = y + dy[k];

        if (nx < 0 || ny < 0 || nx >= width || ny >= height)
            continue;

        int nidx = ny * width + nx;
        if (mask[nidx] == -1)
            continue;

        if ((nx / tile_size) != tile_x || (ny / tile_size) != tile_y)
            continue;

        // Lees ouders
        int pa = parent[idx];
        int pb = parent[nidx];

        if (pa == pb)
            continue;

        // Convergeer naar de kleinste root
        int new_root = min(pa, pb);

        atomic_min(&parent[pa], new_root);
        atomic_min(&parent[pb], new_root);
        atomic_min(&parent[idx], new_root);
        atomic_min(&parent[nidx], new_root);
    }
}
