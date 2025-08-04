__kernel void flatten_roots(__global const int* parent,
                             __global int* labels,
                             const int num_pixels) {
    int gid = get_global_id(0);
    if (gid >= num_pixels)
        return;

    int x = gid;
    if (parent[x] == -1) {
        labels[x] = -1;
        return;
    }

    // Zoek root
    while (parent[x] != x) {
        x = parent[x];
    }

    labels[gid] = x;
}
