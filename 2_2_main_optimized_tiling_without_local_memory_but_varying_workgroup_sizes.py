import time
from PIL import Image
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

def union_find_tiled(image, tile_size, workgroup_size):    
    height, width, channels = image.shape

    if channels == 3:
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        image = np.concatenate((image, alpha), axis=2)

    img_flat = image.reshape((-1, 4))
    num_pixels = width * height

    # OpenCL setup
    platform = cl.get_platforms()[0]
    device = platform.get_devices(cl.device_type.GPU)[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    mf = cl.mem_flags

    # Buffers
    img_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_flat)
    mask_buf = cl.Buffer(context, mf.READ_WRITE, size=4 * num_pixels)
    parent_buf = cl.Buffer(context, mf.READ_WRITE, size=4 * num_pixels)
    label_buf = cl.Buffer(context, mf.WRITE_ONLY, size=4 * num_pixels)
    changes_buf = cl.Buffer(context, mf.READ_WRITE, size=4)
    
    # Helpermethodetje om een kernel te builden
    def build_kernel(fname):
        with open(os.path.join("kernels", fname)) as f:
            return cl.Program(context, f.read()).build()
        
    build_kernel("threshold_mask.cl").threshold_mask(queue, (width, height), None, img_buf, mask_buf, np.int32(width), np.int32(height), np.int32(THRESHOLD))
    build_kernel("initialize_union_data.cl").initialize_union(queue, (num_pixels,), None, mask_buf, parent_buf, np.int32(num_pixels))

    kernel_union_within_tile = build_kernel("union_within_tile.cl")
    kernel_union_horizontal_borders = build_kernel("union_horizontal_borders.cl")
    kernel_union_vertical_borders = build_kernel("union_vertical_borders.cl")

    # De globale grootte moet een veelvoud zijn van de workgroup size, dus hier voeg ik padding toe indien nodig.
    # Nodig om te voorkomen dat er out of bounds errors zijn. We voegen gewoon extra 'dummy pixels' toe om tot een correct veelvoud te komen van de workgroup size.
    workgroupsize_x, workgroupsize_y = workgroup_size
    g_width_padded = (width + workgroupsize_x - 1) // workgroupsize_x * workgroupsize_x
    g_height_padded = (height + workgroupsize_y - 1) // workgroupsize_y * workgroupsize_y
    global_work_shape_2d_main = (g_width_padded, g_height_padded)
    
    # Iteratieve union join totdat er geen wijzigingen meer gedetcteerd worden.
    while True:
        cl.enqueue_copy(queue, changes_buf, np.zeros(1, dtype=np.int32))

        kernel_union_within_tile.union_within_tile(
            queue, global_work_shape_2d_main, workgroup_size,
            mask_buf, parent_buf, np.int32(width), np.int32(height),
            np.int32(tile_size), changes_buf
        )
        
        border_rows = (height - 1) // tile_size
        if border_rows > 0:
            # Pad de 'width' dimensie voor de horizontale grens-kernel
            padded_width = (width + workgroupsize_x - 1) // workgroupsize_x * workgroupsize_x
            horizontal_shape = (padded_width, border_rows)
            horizontal_workgroupsize = (workgroupsize_x, 1)
            kernel_union_horizontal_borders.union_horizontal_borders(
                queue, horizontal_shape, horizontal_workgroupsize,
                mask_buf, parent_buf, np.int32(width), np.int32(height),
                np.int32(tile_size), changes_buf
            )

        border_cols = (width - 1) // tile_size
        if border_cols > 0:
            # Pad de 'height' dimensie voor de verticale grens-kernel
            padded_height = (height + workgroupsize_y - 1) // workgroupsize_y * workgroupsize_y
            vertical_shape = (border_cols, padded_height)
            vertical_workgroupsize = (1, workgroupsize_y)
            kernel_union_vertical_borders.union_vertical_borders(
                queue, vertical_shape, vertical_workgroupsize,
                mask_buf, parent_buf, np.int32(width), np.int32(height),
                np.int32(tile_size), changes_buf
            )

        host_changes = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(queue, host_changes, changes_buf)
        queue.finish()
        
        if host_changes[0] == 0:
            break
            
    build_kernel("flatten_roots.cl").flatten_roots(
        queue, (num_pixels,), None, parent_buf, label_buf, np.int32(num_pixels)
    )

    label_host = np.empty(num_pixels, dtype=np.int32)
    cl.enqueue_copy(queue, label_host, label_buf).wait()
    label_matrix = label_host.reshape((height, width))

    return label_matrix, context, queue


def main():
    # Workgroup size doet niet veel. Het programma is vooral memory bound. We kunnen wel grotere of kleinere workgroups maken, maar aangezien
    # we geen local memory gebruiken zal de impact hiervan heel beperkt zijn.
    work_group_sizes_to_test = [(8, 8)]#, (16, 8), (16, 16), (32, 16), (32, 32)]
    
    for work_group_size in work_group_sizes_to_test:
        wgs_str = f"{work_group_size[0]}x{work_group_size[1]}"
        print(f"Threshold: {THRESHOLD}")

        for image_path in IMAGES:
            image_name = image_path.rsplit("/", maxsplit=1)[-1].split(".")[0]
            output_file = f"perf_{image_name}_wgs_{wgs_str}_.txt"
            with open(output_file, "w") as f_out:
                runs = range(1)
                for run in runs:
                    print()
                    print(f"Image: {image_name} ({image_path})")

                    # Load image
                    img = load_image(image_path)
                    img_arr = np.asarray(img).astype(np.uint8)
                    print(f"Image size: {img_arr.shape}, {img_arr.size} pixels")
                    tile_size = 16


                    start_time = time.perf_counter()

                    # Verwerk image
                    label_matrix, context, queue = union_find_tiled(img_arr, tile_size, work_group_size) # De workgroupsize nu meegeven

                    # Aantal unieke componenten tellen
                    unique_labels = np.unique(label_matrix[label_matrix != -1])
                    cell_count = len(unique_labels)


                    end_time = time.perf_counter()
                    elapsed = end_time - start_time
                    print(f"Cell count: {cell_count}")
                    print(f"Execution time: {elapsed:.4f}s")
                    f_out.writelines(f"{elapsed:.4f}\n")
                    print(f"Run {run+1}/{runs}: {elapsed:.4f}s")

                    # # Voor visualisatie
                    # cell_image = highlight_cells(label_matrix)
                    # Image.fromarray(cell_image).save(f"{image_name}.result.png")


if __name__ == "__main__":
    main()
