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



def union_find_tiled(image, tile_size):    
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
    img_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_flat) #inputafbeelding
    mask_buf    = cl.Buffer(context, mf.READ_WRITE, size=4 * num_pixels) # buffer voor de mask die voor elke pixel -1 of het pixelid bevat
    parent_buf  = cl.Buffer(context, mf.READ_WRITE, size=4 * num_pixels) # Houdt bij naar welke parent een pixel verwijst.
    label_buf   = cl.Buffer(context, mf.WRITE_ONLY, size=4 * num_pixels) # Bevat de uiteindelijke root van een pixel
    changes_buf = cl.Buffer(context, mf.READ_WRITE, size=4) # Houdt bij of er tijdens de iteratie wijzigingen zijn gebeurt. Dit gebruiken we om na te gaan of we outer while loop nog eens moeten oproepen om verder te joinen.

    # Helpermethodetje om een kernel te builden
    def build_kernel(fname):
        with open(os.path.join("kernels", fname)) as f:
            return cl.Program(context, f.read()).build()

    # Kernel threshold_mask.cl
    build_kernel("threshold_mask.cl").threshold_mask(
        queue, (width, height), None,
        img_buf, mask_buf, np.int32(width), np.int32(height), np.int32(THRESHOLD)
    )

    # Kernel initialize_union.cl
    build_kernel("initialize_union_data.cl").initialize_union(
        queue, (num_pixels,), None,
        mask_buf, parent_buf, np.int32(num_pixels)
    )

    kernel_union_within_tile = build_kernel("union_within_tile.cl")
    kernel_union_horizontal_borders = build_kernel("union_horizontal_borders.cl")
    kernel_union_vertical_borders = build_kernel("union_vertical_borders.cl")

    
    # Iteratieve union join totdat er geen wijzigingen meer gedetcteerd worden.
    while True:
        cl.enqueue_copy(queue, changes_buf, np.zeros(1, dtype=np.int32))

        # Union find binnenin een tile
        kernel_union_within_tile.union_within_tile(
            queue, (width, height), None,
            mask_buf, parent_buf, np.int32(width), np.int32(height),
            np.int32(tile_size), changes_buf
        )

        # Nu doen we de union tussen de verschillende tiles op de randen (verticaal en horizontaal langs de randen van de tiles)
        kernel_union_horizontal_borders.union_horizontal_borders(
            queue, (width, height), None,
            mask_buf, parent_buf, np.int32(width), np.int32(height),
            np.int32(tile_size), changes_buf
        )
        kernel_union_vertical_borders.union_vertical_borders(
            queue, (width, height), None,
            mask_buf, parent_buf, np.int32(width), np.int32(height),
            np.int32(tile_size), changes_buf
        )

        host_changes = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(queue, host_changes, changes_buf)
        queue.finish()
        
        if host_changes[0] == 0:
            break
            
    # Kernel flatten_roots.cl
    build_kernel("flatten_roots.cl").flatten_roots(
        queue, (num_pixels,), None,
        parent_buf, label_buf, np.int32(num_pixels)
    )

    # Resultaten terug naar host kopiëren zodat we deze kunnen uitlezen.
    label_host = np.empty(num_pixels, dtype=np.int32)
    cl.enqueue_copy(queue, label_host, label_buf)
    queue.finish()
    label_matrix = label_host.reshape((height, width))

    return label_matrix, context, queue


def main():
    # De tile sizes maken niet uit aangezien de gegevens van een tile niet naar local memory worden gekopieerd en men dus op global memory blijft werken.
    # Hierdoor zal de tile grootte niet uitmaken omdat men dus nooit meer gegevens naar local memory zal kopiëren bij een hogere tile size en er dus ook geen snellere geheugenacces is.
    # Het programma is memory bound
    for tile_size in [4]:#, 8, 16, 32, 64]:
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
                    # cell_image = highlight_cells(label_matrix)
                    # Image.fromarray(cell_image).save(f"{image_name}.result.png")

                    end_time = time.perf_counter()
                    elapsed = end_time - start_time
                    print(f"Cell count: {cell_count}")
                    print(f"Execution time: {elapsed:.4f}s")

                    f_out.writelines(f"{elapsed:.4f}\n")
                    print(f"Run {run+1}/{runs}: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
