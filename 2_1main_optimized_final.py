import time
from PIL import Image
import argparse
import numpy as np
import pyopencl as cl
import os


# A pixel is part of a cell if its brightness is lower than THRESHOLD.
THRESHOLD = 200

IMAGES = [
    "./images/Anisocytose_plaquettaire.jpg",
    "./images/Eosinophil_blood_smear.jpg",
    "./images/Neutrophils_monocyte_16694967012.jpg",
    "./images/Patology_of_platelet.jpg",
    "./images/Plaquettes_normales.jpg",
    "./images/Plaquette_geante.jpg",
    "./images/Platelets2.jpg",
    "./images/RBCs_Platelets_WBC_in_PBS_2.jpg",
    "./images/Thrombocytes.jpg",
    "./images/Thrombocytosis.jpg",
    "./images/tiny-example.bmp",  # Used in the assignment
]


def load_image(path):
    """Load an image file into a PIL image."""
    return Image.open(path)


def pixel_brightness(pixel):
    """Calculate the brightness of a pixel."""
    # The brightness of a pixel is the average of its RGB values.
    # Note that pixels are of type np.uint8 (0-255), so if we just sum them, we
    # get overflows. Hence, we convert them to np.float32 first.
    # Note also that rounding in Python may be different from rounding in
    # OpenCL (also depending on if/where you convert to float or another int
    # size). You can adjust the Python code below to match your OpenCL code.
    return np.sum(pixel.astype(np.float32)) / 3


def is_part_of_cell(pixel):
    """Check if a pixel is part of a cell, i.e. if its brightness is below the
    threshold."""
    return pixel_brightness(pixel) < THRESHOLD


def union_find_tiled(image, tile_size):    
    height, width, channels = image.shape

    assert channels in [3, 4], "Image must have 3 (RGB) or 4 (RGBA) channels."
    if channels == 3:
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        image = np.concatenate((image, alpha), axis=2)

    img_flat = image.reshape((-1, 4))
    num_pixels = width * height
    THRESHOLD = 200

    # OpenCL setup
    platform = cl.get_platforms()[0]
    device = platform.get_devices(cl.device_type.GPU)[0]

    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    mf = cl.mem_flags

    # Buffers
    img_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_flat)
    mask_buf    = cl.Buffer(context, mf.READ_WRITE, size=4 * num_pixels)
    parent_buf  = cl.Buffer(context, mf.READ_WRITE, size=4 * num_pixels)
    rank_buf    = cl.Buffer(context, mf.READ_WRITE, size=4 * num_pixels)
    label_buf   = cl.Buffer(context, mf.WRITE_ONLY, size=4 * num_pixels)
    changes_buf = cl.Buffer(context, mf.READ_WRITE, size=4)

    def build_kernel(fname):
        with open(os.path.join("kernels", fname)) as f:
            return cl.Program(context, f.read()).build()

    # -------- KERNEL 1: threshold_mask --------
    build_kernel("threshold_mask.cl").threshold_mask(
        queue, (width, height), None,
        img_buf, mask_buf, np.int32(width), np.int32(height), np.int32(THRESHOLD)
    )

    # -------- KERNEL 2: initialize_union --------
    build_kernel("initialize_union_data.cl").initialize_union(
        queue, (num_pixels,), None,
        mask_buf, parent_buf, rank_buf, np.int32(num_pixels)
    )

    kernel_union_within_tile    = build_kernel("union_within_tile.cl")
    kernel_union_across_borders = build_kernel("union_across_borders.cl")
    # kernel_union_vertical       = build_kernel("union_vertical_borders.cl")
    # kernel_union_horizontal     = build_kernel("union_horizontal_borders.cl")
    # kernel_union_diagonal       = build_kernel("union_diagonal_borders.cl")

    # -------- ITERATIEVE UNION --------
    while True:
        cl.enqueue_copy(queue, changes_buf, np.zeros(1, dtype=np.int32))

        # Stap 1: Verbind pixels BINNEN de tiles
        kernel_union_within_tile.union_within_tile(
            queue, (width, height), None,
            mask_buf, parent_buf, np.int32(width), np.int32(height),
            np.int32(tile_size), changes_buf
        )

        # Stap 2: Verbind pixels OVER de grenzen van de tiles
        kernel_union_across_borders.union_across_borders(
            queue, (width, height), None,
            mask_buf, parent_buf, np.int32(width), np.int32(height),
            np.int32(tile_size), changes_buf
        )

        host_changes = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(queue, host_changes, changes_buf)
        queue.finish()
        if host_changes[0] == 0:
            break
    
    # --- CORRECTIE: Deze code staat nu NA de while-lus ---

    # -------- KERNEL 6: flatten_roots --------
    build_kernel("flatten_roots.cl").flatten_roots(
        queue, (num_pixels,), None,
        parent_buf, label_buf, np.int32(num_pixels)
    )

    # Kopieer naar host
    label_host = np.empty(num_pixels, dtype=np.int32)
    cl.enqueue_copy(queue, label_host, label_buf)
    queue.finish() # Wacht op de kopieer-operatie
    label_matrix = label_host.reshape((height, width))

    return label_matrix, context, queue



def highlight_cells(cell_numbers):
    """Generate an image with the cells highlighted.

    This is not needed for the algorithm, but can be useful for debugging."""
    height, width = cell_numbers.shape
    cell_count = cell_numbers.max() + 2
    # +2 because the background is -1, and we want to include the last cell
    # number in the range.

    # Set random seed to get the same colors each time
    np.random.seed(0)

    # Generate a random color for each cell.
    # This is an matrix of shape (cell_count, 3), where each row is a color.
    # E.g. colors[123] is the RGB color for cell 123.
    # We use colors from 0 to 200 to avoid very light colors (close to white).
    colors = np.random.randint(0, 200, (cell_count, 3), dtype=np.uint8)
    # Cell number -1 is reserved for background pixels that are not part of a
    # cell: make these pixels white.
    colors[-1] = [255, 255, 255]

    # Create an image to store the result
    result = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign a color to each pixel based on the cell number
    for row in range(height):
        for col in range(width):
            result[row, col] = colors[cell_numbers[row, col]]
    return result


def matrix_to_svg(matrix, filename):
    """Convert a matrix of numbers to an SVG file.

    This is for the purposes of debugging. I used this to generate SVG files
    for the assignment. Note that this will only work for small images,
    otherwise the SVG file becomes too large.
    """
    height, width = matrix.shape
    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 {width} {height}">\n'
        )
        f.write(
            "<style>\nrect { fill: black; }\ntext { fill: white; font-size: 0.5px; text-anchor: middle; dominant-baseline: middle; }\n</style>\n"
        )
        for row in range(height):
            for col in range(width):
                if matrix[row, col] == -1:
                    continue
                f.write(f'<rect x="{col}" y="{row}" width="1" height="1" />\n')
                f.write(f'<text x="{col}.5" y="{row}.5">{matrix[row, col]}</text>\n')
        f.write("</svg>\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="Aantal benchmark runs")
    parser.add_argument("--tilesize", type=int, default=16, help="Tile size (optioneel)")
    parser.add_argument("--workgroupsize", type=str, help="bv. 16x16 (optioneel)")
    parser.add_argument("--output", type=str, required=True, help="Output bestand om runtimes te loggen")
    return parser.parse_args()

def parse_workgroup_size(s):
    if s is None:
        return None
    parts = s.lower().split("x")
    return tuple(int(p) for p in parts)


def main():
    # #dragonfly_cores = 16384;
    # #personalpc_cores = 2176;

    # dragonfly_workgroupsizes = [(64, 64), (64, 128), (128, 128),(128,256), (256, 256), (256, 512), (512, 512)]
    # personalpc_workgroupsizes = [(32, 32), (64, 64), (128, 64)]


    # ideal_workgroupsize_personalpc = (128, 64)
    # ideal_workgroupsize_dragonfly = (256, 256)

    # ideal_workgroupsize = ideal_workgroupsize_personalpc
 
    # workgroupsizes = personalpc_workgroupsizes

    # testcases = [
    #     ((32, 32), (256,)),
    #     ((64, 64), (512,)),
    #     ((128, 64), (1024,))
    # ]

    for tile_size in [4]:#, 8, 16, 32, 64, 128]:
        print(f"Threshold: {THRESHOLD}")

        for image_path in IMAGES:
            image_name = image_path.rsplit("/", maxsplit=1)[-1].split(".")[0]
            output_file = f"2_1_{image_name}_tilsize_{tile_size}_.txt"
            with open(output_file, "w") as f_out:
                runs = range(1)
                for run in runs:
                    print()
                    print(f"Image: {image_name} ({image_path})")

                    # Load image
                    img = load_image(image_path)
                    img_arr = np.asarray(img).astype(np.uint8)
                    print(f"Image size: {img_arr.shape}, {img_arr.size} pixels")

                    start_time = time.perf_counter()

                    # Verwerk image
                    label_matrix, context, queue = union_find_tiled(img_arr, tile_size)

                    # Aantal unieke componenten tellen
                    unique_labels = np.unique(label_matrix[label_matrix != -1])
                    cell_count = len(unique_labels)

                    # Voor visualisatie
                    cell_image = highlight_cells(label_matrix)
                    Image.fromarray(cell_image).save(f"{image_name}.result.png")

                    end_time = time.perf_counter()
                    elapsed = end_time - start_time
                    print(f"Cell count: {cell_count}")
                    print(f"Execution time: {elapsed:.4f}s")

                    f_out.writelines(f"{elapsed:.4f}\n")
                    print(f"Run {run+1}/{runs}: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
