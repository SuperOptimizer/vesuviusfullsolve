


def visualize_chords(zchords, ychords, xchords):
    """Fallback visualization using chords when no patches are generated"""
    all_centroids = []
    all_values = []
    all_indices = []

    for i, chord_set in enumerate([zchords, ychords, xchords]):
        for j, chord_ in enumerate(chord_set):
            points = [(p.z, p.y, p.x) for p in chord_]
            values = [p.c for p in chord_]
            indices = [j%1024] * len(chord_)

            all_centroids.extend(points)
            all_values.extend(values)
            all_indices.extend(indices)

    if not all_centroids:
        raise ValueError("No visualization data available")

    return all_centroids, all_values, all_indices


def visualize_paths(zpaths, ypaths, xpaths):
    """Visualize the longer continuous paths with different colors for each direction"""
    all_centroids = []
    all_values = []
    all_indices = []

    # Assign different base indices for each direction to create distinct color ranges
    direction_offsets = {
        'z': 0,  # Will use viridis colormap
        'y': 256,  # Will use magma colormap
        'x': 512  # Will use inferno colormap
    }

    for direction, paths in [('z', zpaths), ('y', ypaths), ('x', xpaths)]:
        base_idx = direction_offsets[direction]

        for i, path in enumerate(paths):
            # Ensure we don't exceed color index limits
            path_idx = base_idx + (i % 256)

            # Extract points from path
            points = [(p.z, p.y, p.x) for p in path]
            values = [p.c for p in path]
            indices = [path_idx] * len(path)

            all_centroids.extend(points)
            all_values.extend(values)
            all_indices.extend(indices)

    if not all_centroids:
        raise ValueError("No visualization data available")

    return all_centroids, all_values, all_indices