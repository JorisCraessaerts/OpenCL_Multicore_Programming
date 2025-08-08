__kernel void threshold_mask(__global const uchar4* img,
                             __global int* mask,
                             const int width,
                             const int height,
                             const int threshold) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Als we buiten de genszen gaan.
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    uchar4 pixel = img[idx];
    float brightness = (pixel.x + pixel.y + pixel.z) / 3.0f;

    mask[idx] = select(-1, idx, brightness < threshold); // Hiermee vermijden we de branch divergence die door de if else werd veroorzaakt

}