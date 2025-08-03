#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2)

__kernel void union_find_tiled(__global int* parent,
                                __global const int* mask,
                                __global int* changed,
                                const int width,
                                const int height) {
    __local int tile[TILE_SIZE][TILE_SIZE];

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    int gidx = gy * width + gx;

    // 1. Local tile-coord (lx, ly) ↔ global (gx, gy)
    // Bereken tile origin in global geheugen
    int tile_gx = gx - lx + max(0, lx - 1);
    int tile_gy = gy - ly + max(0, ly - 1);

    // 2. Global coördinaten van dit punt binnen tile, met rand
    int tx = lx + 1;
    int ty = ly + 1;

    // 3. Laad pixel én z’n rand in local memory
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int local_x = tx + dx;
            int local_y = ty + dy;
            int global_x = gx + dx;
            int global_y = gy + dy;

            if (local_x >= 0 && local_x < TILE_SIZE &&
                local_y >= 0 && local_y < TILE_SIZE &&
                global_x >= 0 && global_x < width &&
                global_y >= 0 && global_y < height) {
                int gid = global_y * width + global_x;
                tile[local_y][local_x] = mask[gid];
            } else {
                tile[local_y][local_x] = -1;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // 4. Main WI werkt op (lx, ly) → (tx, ty)
    if (gx >= width || gy >= height) return;
    if (tile[ty][tx] == -1) return;

    int this_id = gy * width + gx;

    int neighbors[4][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}
    };

    for (int i = 0; i < 4; i++) {
        int nx = tx + neighbors[i][1];
        int ny = ty + neighbors[i][0];

        if (nx < 0 || nx >= TILE_SIZE || ny < 0 || ny >= TILE_SIZE)
            continue;

        if (tile[ny][nx] == -1)
            continue;

        int neighbor_global_id = (gy + neighbors[i][0]) * width + (gx + neighbors[i][1]);

        int p1 = parent[this_id];
        int p2 = parent[neighbor_global_id];

        if (p1 < p2) {
            if (atomic_min(&parent[p2], p1) != p1)
                *changed = 1;
        } else if (p2 < p1) {
            if (atomic_min(&parent[p1], p2) != p2)
                *changed = 1;
        }
    }
}
