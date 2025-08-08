__kernel void initialize_union(__global int* mask,
                               __global int* parent,
                               const int size) {
    int gid = get_global_id(0);
    if (gid >= size) return;

    parent[gid] = select(gid, -1, mask[gid] == -1); // parent id zetten op basis van of er -1 zit in de mask. Ik gebruik hier select om branch divergence te vermijden.
}