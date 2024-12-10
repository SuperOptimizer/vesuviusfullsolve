import sys


def interpolate_coordinates(input_lines, step=2):
    # First pass - get all explicitly given coordinates
    coords = []
    for line in input_lines:
        parts = line.strip().split(", ")
        z = int(parts[0].split(": ")[1])
        x = int(parts[1].split(": ")[1])
        y = int(parts[2].split(": ")[1])
        coords.append((z * step, x * step, y * step))  # Scale by step size

    # Sort by z coordinate
    coords.sort()

    # Initialize output list with None placeholders for all z values
    max_z = coords[-1][0]
    output = [None] * (max_z + 1)

    # Place known coordinates
    for z, x, y in coords:
        output[z] = (x, y)

    # Interpolate missing values between each pair of known points
    for base_z in range(0, max_z, step):
        if base_z + step >= len(output):
            break

        # Only interpolate if we have both endpoints
        if output[base_z] is not None and output[base_z + step] is not None:
            start_x, start_y = output[base_z]
            end_x, end_y = output[base_z + step]

            # Interpolate all points between base_z and base_z + step
            for i in range(1, step):
                # Calculate interpolation factor
                factor = i / step

                # Linear interpolation
                interp_x = int(start_x + (end_x - start_x) * factor)
                interp_y = int(start_y + (end_y - start_y) * factor)

                output[base_z + i] = (interp_x, interp_y)

    # Output results
    print("z,y,x")  # Header
    for z, coords in enumerate(output):
        if coords is not None:  # Skip any wwNone values at the end
            x, y = coords
            print(f"{z},{y},{x}")


if __name__ == "__main__":
    # Example usage with different step sizes
    filename = "scroll1b_2_umbilicus.txt"
    with open(filename, 'rt') as f:
        lines = f.readlines()
        interpolate_coordinates(lines, step=4)  # Change step size as needed