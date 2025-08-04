__kernel void transpose_int_buffer(__global const int* input,
                                   __global int* output,
                                   const int width,
                                   const int height) {
    int x = get_global_id(0); // kolom
    int y = get_global_id(1); // rij

    if (x < width && y < height) {
        int in_index  = y * width + x;
        int out_index = x * height + y; // transpositie!
        output[out_index] = input[in_index];
    }
}