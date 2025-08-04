__kernel void initialize_union(__global int* mask,
                               __global int* parent,
                               __global int* rank,
                               const int size) {
    int gid = get_global_id(0);
    if (gid >= size) return;

    if (mask[gid] == -1) {
        parent[gid] = -1;
    } else {
        parent[gid] = gid;
    }
    rank[gid] = 0;
}