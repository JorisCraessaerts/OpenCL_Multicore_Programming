// Veilige flatten_roots.cl met path compression
__kernel void flatten_roots(__global int* parent,
                            __global int* labels,
                            const int num_pixels) {
    int gid = get_global_id(0);
    if (gid >= num_pixels) return;

    int current_pixel = gid;
    if (parent[current_pixel] == -1) {
        labels[current_pixel] = -1;
        return;
    }

    // Stap 1: Vind de uiteindelijke root met een veilige lus
    int root = current_pixel;
    while (parent[root] != root && parent[root] != -1) {
        root = parent[root];
    }
    
    // Als de keten naar een achtergrondpixel leidde, markeer als achtergrond.
    if (root == -1) {
        labels[current_pixel] = -1;
        return;
    }

    // Stap 2: Pas path compression toe. Dit is hier veilig, want elke thread
    // bewerkt alleen de keten die bij zijn EIGEN unieke 'gid' hoort.
    int temp_pixel = current_pixel;
    while (parent[temp_pixel] != root) {
        int old_parent = parent[temp_pixel];
        parent[temp_pixel] = root;
        temp_pixel = old_parent;
    }

    labels[current_pixel] = root;
}