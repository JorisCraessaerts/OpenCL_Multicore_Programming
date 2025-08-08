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

    // 1) Vind de uiteindelijke root
    int root = current_pixel;
    while (parent[root] != root && parent[root] != -1) {
        root = parent[root];
    }
    
    // Als root een achtergrondpixel zou zijn, zet dan ook de label van deze pixel naar die van een achtergrondpixel
    if (root == -1) {
        labels[current_pixel] = -1;
        return;
    }

    // 2) Pas path compression toe. Pixels kunnen wel buiten hun tile grenzen gaan, maar dit vormt geen probleem zolang men met global memory werkt. Wanneer men met local memory werkt, geeft dit problemen. Nog geen oplossing voor gevonden.
    int temp_pixel = current_pixel;
    while (parent[temp_pixel] != root) {
        int old_parent = parent[temp_pixel];
        parent[temp_pixel] = root;
        temp_pixel = old_parent;
    }

    labels[current_pixel] = root;
}