// Vereenvoudigde, veilige find_root ZONDER path compression
inline int find_root(__global int* parent, int i) {
    if (i == -1) {
        return -1;
    }
    int root = i;
    while (parent[root] != root && parent[root] != -1) {
        root = parent[root];
    }
    return root;
}

// Deze kernel vervangt de 3 oude grens-kernels.
// Hij wordt aangeroepen met een 2D global size (width, height).
__kernel void union_across_borders(__global const int* mask,
                                   __global int* parent,
                                   const int width,
                                   const int height,
                                   const int tile_size,
                                   __global int* changes_made) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // Sla achtergrondpixels over.
    int idx = y * width + x;
    if (mask[idx] == -1) return;

    // We controleren alleen buren voor pixels die AAN de rand van een tile liggen.
    // Dit is een optimalisatie om niet elke pixel te moeten controleren.
    bool on_border = (x % tile_size == 0 && x > 0) || (x % tile_size == tile_size - 1 && x < width - 1) ||
                     (y % tile_size == 0 && y > 0) || (y % tile_size == tile_size - 1 && y < height - 1);

    if (!on_border) return;

    int root_a = find_root(parent, idx);
    if (root_a == -1) return;

    // Controleer de 4 "forward" buren (rechts, rechtsonder, onder, linksonder)
    // Dit is voldoende omdat elke grens dan vanuit één kant wordt gecontroleerd.
    const int dx[4] = {1, 1, 0, -1};
    const int dy[4] = {0, 1, 1,  1};

    for (int k = 0; k < 4; k++) {
        int nx = x + dx[k];
        int ny = y + dy[k];

        // Controleer of de buur binnen de afbeelding is
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            int nidx = ny * width + nx;

            if (mask[nidx] != -1) {
                int root_b = find_root(parent, nidx);

                if (root_b != -1 && root_a != root_b) {
                    // Veilige, atomaire union-operatie
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