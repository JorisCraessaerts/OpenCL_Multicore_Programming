import time
from PIL import Image
import numpy as np

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


def union_find_tiled(image):
    """Convert an image to a "mask matrix". In this matrix:
    - Pixels that are not part of a cell (= "background pixels") are set to -1.
    - Pixels that are part of a cell are set to a unique number. This number is
      the 'global index' of the pixel, i.e. `row * width + col`.
    """
    height, width, _channels = image.shape  # channels is 3 for RGB images
    mask_matrix = np.zeros((height, width), dtype=np.int32)

    for row in range(height):  # row = y
        for col in range(width):  # col = x
            pixel = image[row, col]
            if is_part_of_cell(pixel):
                mask_matrix[row, col] = row * width + col
            else:
                mask_matrix[row, col] = -1

    return mask_matrix


def count_cells(mask_matrix):
    """Find and count the number of cells in the mask matrix.
    This returns the number of cells, as well as an 'image' of cell numbers."""
    height, width = mask_matrix.shape

    # The maximum number of cells we could ever have is the number of pixels.
    max_cells = height * width

    # The algorithm we use is called "union find" or "union by rank". You can
    # read more about it at
    # https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    # The parent array keeps track of the parent of each pixel. If a pixel is
    # its own parent, it is a "root pixel".
    # Note: np.arange(n) creates an array [0, 1, 2, ..., n], so each pixel
    # starts as a root pixel.
    parent = np.arange(max_cells, dtype=np.int32)
    # Rank per pixel, initialized to 0.
    rank = np.zeros(max_cells, dtype=np.int32)

    def find(x):
        """Find the root of a pixel."""
        if parent[x] != x:  # Is this a root pixel?
            # No: continue up the tree
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        """Merge two pixels and their cells."""
        # Find the root of each pixel
        rootX = find(x)
        rootY = find(y)

        if rootX != rootY:
            # If the roots are different, merge the cells
            if rank[rootX] > rank[rootY]:
                # X has a higher rank: make Y a child of X
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                # Y has a higher rank: make X a child of Y
                parent[rootX] = rootY
            else:
                # Both have the same rank: make one a child of the other and
                # increase the rank of the new parent.
                parent[rootY] = rootX
                rank[rootX] += 1

    # Iterate over all pixels, and merge neighboring pixels into the same cell.
    for row in range(height):
        for col in range(width):
            if mask_matrix[row, col] == -1:
                # Background pixel: skip
                continue

            # Check if the neighbors are part of a cell, and merge with these.
            # Note: we only need to check the previous (top-left, top,
            # top-right, and left) neighbors; the later ones (right,
            # bottom-left, bottom, bottom-right) will check this pixel in their
            # iteration.
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                r = row + dr
                c = col + dc

                # Skip if the neighbor is out of bounds, or a background pixel
                if r < 0 or r >= height:
                    continue
                if c < 0 or c >= width:
                    continue
                if mask_matrix[r, c] == -1:
                    continue

                # Merge the two pixels' cells
                union(mask_matrix[row, col], mask_matrix[r, c])

    # Create an array to store the cell numbers
    # Note that cell numbers are not necessarily contiguous.
    cell_numbers = np.zeros((height, width), dtype=np.int32)

    # Update the cell numbers with the root numbers
    for row in range(height):
        for col in range(width):
            if mask_matrix[row, col] == -1:
                cell_numbers[row, col] = -1
            else:
                cell_numbers[row, col] = find(mask_matrix[row, col])

    # Count the number of cells
    # Subtract 1 to ignore the background value
    cell_count = len(np.unique(cell_numbers)) - 1

    # Return the number of cells, as well as the 'image' of cell numbers.
    # The cell numbers can be used for debugging.
    return (cell_count, cell_numbers)


def count_cells_flood_fill(mask_matrix):
    """Find and count the number of cells in the mask matrix using a flood
    fill algorithm.

    This is a simpler algorithm than the union find algorithm above, and will
    typcially be faster in a sequential execution. However, it is not as easy
    to parallelize on a GPU, because it requires either a queue or recursion.
    """
    height, width = mask_matrix.shape

    # The cell number for each pixel. -1 means the pixel is not part of a cell.
    # Initialize with all -1.
    cell_numbers = np.full((height, width), -1, dtype=np.int32)

    # The next cell number to use
    cell_index = 0

    def flood_fill(row, col, cell_number):
        """Fill a cell using a flood fill algorithm."""
        queue = [(row, col)]
        while len(queue) > 0:
            row, col = queue.pop(0)

            if row < 0 or row >= height:
                continue
            if col < 0 or col >= width:
                continue
            if mask_matrix[row, col] == -1:
                # Background pixel: skip
                continue
            if cell_numbers[row, col] != -1:
                # Already visited by a previous flood fill
                continue

            cell_numbers[row, col] = cell_number

            for dr, dc in [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                # skip self
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]:
                queue.append((row + dr, col + dc))

    # Iterate over all pixels, and fill each cell using a flood fill.
    for row in range(height):
        for col in range(width):
            if mask_matrix[row, col] == -1:
                continue

            if cell_numbers[row, col] != -1:
                # Already part of a cell: skip
                continue
            # Not part of a cell yet: fill this cell
            flood_fill(row, col, cell_index)
            cell_index += 1

    return (cell_index, cell_numbers)


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
        
def tiled_union_find(mask_matrix, tile_size=16):
    height, width = mask_matrix.shape
    num_pixels = height * width

    parent = np.full(num_pixels, -1, dtype=np.int32)

    def index(y, x):
        return y * width + x

    # Dit is gewoon de find om de root te vinden. We blijven zoeken zolang de parent niet gelijk is aan de huidige pixel of niet gelijk is aan een achtergrondpixel
    def find(x):
        while parent[x] != x and parent[x] != -1:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    # Union om twee pixels aan elkaar te hangen.
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == -1 or root_y == -1 or root_x == root_y:
            return False
        if root_x < root_y:
            parent[root_y] = root_x
        else:
            parent[root_x] = root_y
        return True

    # Initializeer de parent parent: 
    for y in range(height):
        for x in range(width):
            idx = index(y, x) # Zoek de index op op in de ééndimensional rij
            if mask_matrix[y, x] != -1:
                parent[idx] = idx

    # De 8 buren rond de pixel
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),            (0, 1),
                 (1, -1),  (1, 0), (1, 1)]

    changed = True
    while changed:
        changed = False

        # union_within_tiles
        for tile_y in range(0, height, tile_size):
            for tile_x in range(0, width, tile_size):
                for y in range(tile_y, min(tile_y + tile_size, height)): # naar beneden afronden omdat tiles op de randen van de afbeelding natuurlijk minder pixels hebben.
                    for x in range(tile_x, min(tile_x + tile_size, width)): # Hetzelfde hier
                        if mask_matrix[y, x] == -1:
                            continue
                        for dy, dx in neighbors: # Bekijk de directe buren van de tile door de x en y coördinaten overeenkomstig aan te passen.
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if mask_matrix[ny, nx] == -1:
                                    continue
                                if (ny // tile_size, nx // tile_size) == (y // tile_size, x // tile_size):
                                    changed |= union(index(y, x), index(ny, nx))

        # De union over de horizontale grenzen. We gaan horizontaal tussen de tiles lopen en de pixels van de upper en lower tile joinen indien ze overeen komen (verticaal, diagonaal)
        for tile_y in range(1, (height + tile_size - 1) // tile_size):
            y1 = tile_y * tile_size - 1
            y2 = tile_y * tile_size
            if y1 >= height or y2 >= height: continue

            for x in range(width):
                # Controleer de 3 buren in de rij eronder (links-diagonaal, recht onder, rechts-diagonaal)
                for dx in [-1, 0, 1]:
                    nx = x + dx
                    # Zorg ervoor dat de buur binnen de afbeelding valt
                    if 0 <= nx < width:
                        if mask_matrix[y1, x] != -1 and mask_matrix[y2, nx] != -1:
                            changed |= union(index(y1, x), index(y2, nx))

        # De union over de verticale grenzen tussen de tiles. We gaan verticaal tussen de tiles lopen en de pixels van de linker en rechter tile joinen indien ze overeen komen (horizontaal, diagonaal)
        for tile_x in range(1, (width + tile_size - 1) // tile_size):
            x1 = tile_x * tile_size - 1
            x2 = tile_x * tile_size
            if x1 >= width or x2 >= width: continue

            for y in range(height):
                # Controleer enkel de directe buur rechts. De diagonale verbindingen
                # zijn al gedekt door de lus hierboven.
                if mask_matrix[y, x1] != -1 and mask_matrix[y, x2] != -1:
                    changed |= union(index(y, x1), index(y, x2))

    # Step 6: Flatten roots
    label_matrix = np.full((height, width), -1, dtype=np.int32)
    for y in range(height):
        for x in range(width):
            idx = index(y, x)
            if mask_matrix[y, x] != -1 and parent[idx] != -1:
                label_matrix[y, x] = find(idx)

    return label_matrix



def main():
    print(f"Threshold: {THRESHOLD}")

    for image_path in IMAGES:
        image_name = image_path.rsplit("/", maxsplit=1)[-1].split(".")[0]
        print()
        print(f"Image: {image_name} ({image_path})")

        # Load image
        img = load_image(image_path)

        # Convert image to numpy array
        img_arr = np.asarray(img).astype(np.uint8)
        print(f"Image size: {img_arr.shape}, {img_arr.size} pixels")
        # Note: PIL uses (x, y) coordinates, while numpy uses (row, col)
        # coordinates. Hence, the width and height are swapped!
        # print(f"Image shape: {img.size}")
        # print(f"Image array size: {img_arr.shape}")

        # For OpenCL, you can flatten the image to a 1D array using
        # img_arr.flatten()

        start_time = time.perf_counter()

        # 1. Convert the image to a "mask matrix", where True indicates the
        # pixel is part of a cell.
        mask_matrix = union_find_tiled(img_arr)
        # matrix_to_svg(mask_matrix, f"{image_name}.mask.svg")

        # 2. Count the number of cells in the image
        (cell_count, cell_numbers) = count_cells(mask_matrix)
        # matrix_to_svg(cell_numbers, f"{image_name}.result.svg")
        # (cell_count, cell_numbers) = count_cells_flood_fill(mask_matrix)

        end_time = time.perf_counter()

        # Generate image with the cells highlighted, using different colors.
        # This might be useful for debugging.
        cell_image = highlight_cells(cell_numbers)
        Image.fromarray(cell_image).save(f"{image_name}.result.png")
        # Note: even though the input is JPG, it is better to save the output
        # as PNG, to avoid compression artifacts.

        # Display results
        print(f"Cell count: {cell_count}")
        print(f"Execution time: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    main()
