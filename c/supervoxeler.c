#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Constants for 256³ volume
#define D_SEED 2
#define VOLUME_DIM 256
#define VOLUME_SLICE (VOLUME_DIM * VOLUME_DIM)
#define GRID_DIM (VOLUME_DIM / D_SEED)

typedef struct HeapNode {
    uint8_t x, y, z;  // Volume coordinates (0-255)
    uint8_t d;
} HeapNode;

static inline int heap_node_compare(HeapNode a, HeapNode b) {
    // First compare by distance/priority
    int val_diff = b.d - a.d;
    if (val_diff != 0) return val_diff;

    // Break ties using spatial coordinates in alternating dimensions
    int dim_sum = a.x + a.y + a.z;
    switch (dim_sum % 3) {
        case 0:
            if (a.x != b.x) return b.x - a.x;
            if (a.y != b.y) return b.y - a.y;
            return b.z - a.z;
        case 1:
            if (a.y != b.y) return b.y - a.y;
            if (a.z != b.z) return b.z - a.z;
            return b.x - a.x;
        default:
            if (a.z != b.z) return b.z - a.z;
            if (a.x != b.x) return b.x - a.x;
            return b.y - a.y;
    }
}

static inline void get_grid_coords(uint8_t x, uint8_t y, uint8_t z,
                                 uint8_t* grid_x, uint8_t* grid_y, uint8_t* grid_z) {
    *grid_x = x / D_SEED;
    *grid_y = y / D_SEED;
    *grid_z = z / D_SEED;
}

typedef struct Heap {
    int len, size;
    HeapNode* nodes;
} Heap;

typedef struct Supervoxel {
    float x, y, z, c;  // Keep as float for accumulation and averaging
    unsigned int n;
} Supervoxel;

typedef struct SupervoxelGrid {
    Supervoxel* data;  // Flattened 3D array
} SupervoxelGrid;

static inline int grid_index(int grid_x, int grid_y, int grid_z) {
    return grid_z * GRID_DIM * GRID_DIM + grid_y * GRID_DIM + grid_x;
}

static inline Supervoxel* get_supervoxel(SupervoxelGrid* grid, int grid_x, int grid_y, int grid_z) {
    if (grid_x < 0 || grid_x >= GRID_DIM ||
        grid_y < 0 || grid_y >= GRID_DIM ||
        grid_z < 0 || grid_z >= GRID_DIM) {
        return NULL;
    }
    return &grid->data[grid_index(grid_x, grid_y, grid_z)];
}

static inline Heap heap_alloc(int size) {
    size_t alloc_size = (size_t)(size * sizeof(HeapNode) / 2);
    printf("mmap allocating %llu bytes\n", (unsigned long long)alloc_size);

    HeapNode* nodes = (HeapNode*)mmap(
        NULL, alloc_size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
    );

    if (nodes == MAP_FAILED) {
        perror("mmap failed");
        return (Heap){0};
    }

    nodes[0] = (HeapNode){0};
    return (Heap){.len = 0, .size = size / 2, .nodes = nodes};
}

static inline void heap_free(Heap *heap) {
    if (heap->nodes) {
        size_t alloc_size = (size_t)heap->size * sizeof(HeapNode);
        if (munmap(heap->nodes, alloc_size) == -1) {
            perror("munmap failed");
        }
        heap->nodes = NULL;
    }
}

static inline void heap_push(Heap *heap, HeapNode node) {
    int i = ++heap->len;
    HeapNode* nodes = heap->nodes;
    while(i > 1) {
        int parent = i >> 1;
        if(heap_node_compare(nodes[parent], node) >= 0) break;
        nodes[i] = nodes[parent];
        i = parent;
    }
    nodes[i] = node;
}

static inline HeapNode heap_pop(Heap *heap) {
    HeapNode* nodes = heap->nodes;
    HeapNode result = nodes[1];
    HeapNode last = nodes[heap->len--];
    int i = 1;
    while(i <= (heap->len >> 1)) {
        int child = i << 1;
        if(child < heap->len && heap_node_compare(nodes[child], nodes[child + 1]) < 0)
            child++;
        if(heap_node_compare(last, nodes[child]) >= 0) break;
        nodes[i] = nodes[child];
        i = child;
    }
    nodes[i] = last;
    return result;
}

SupervoxelGrid* init_supervoxel_grid(void) {
    SupervoxelGrid* grid = malloc(sizeof(SupervoxelGrid));
    if (!grid) return NULL;

    size_t grid_size = GRID_DIM * GRID_DIM * GRID_DIM;
    grid->data = calloc(grid_size, sizeof(Supervoxel));
    if (!grid->data) {
        free(grid);
        return NULL;
    }

    return grid;
}

void free_supervoxel_grid(SupervoxelGrid* grid) {
    if (grid) {
        free(grid->data);
        free(grid);
    }
}

int supervoxeler(uint8_t *img, SupervoxelGrid* grid) {
    int img_size = VOLUME_DIM * VOLUME_DIM * VOLUME_DIM;
    Heap pq = heap_alloc(img_size*8);

    // Create boolean array to track visited voxels
    uint8_t* visited = calloc(img_size, sizeof(uint8_t));
    if (!visited) {
        heap_free(&pq);
        return -1;
    }

    // Create seeds at grid points
    for (int grid_z = 0; grid_z < GRID_DIM; grid_z++) {
        for (int grid_y = 0; grid_y < GRID_DIM; grid_y++) {
            for (int grid_x = 0; grid_x < GRID_DIM; grid_x++) {
                int x = grid_x * D_SEED;
                int y = grid_y * D_SEED;
                int z = grid_z * D_SEED;
                int idx = (z * VOLUME_SLICE) + (x * VOLUME_DIM) + y;

                if (!visited[idx] && img[idx] > 0) {
                    heap_push(&pq, (HeapNode){
                        .d = 0,
                        .x = x,
                        .y = y,
                        .z = z
                    });
                }
            }
        }
    }

    float invwt = 1.0f / (float)(img_size);
    const float scale = 100.0f;
    const float intensity_weight = 2.0f;

    // Process queue
    while (pq.len > 0) {
        HeapNode n = heap_pop(&pq);
        int i = ((int)n.z * VOLUME_SLICE) + ((int)n.x * VOLUME_DIM) + n.y;

        if (visited[i]) continue;
        visited[i] = 1;

        uint8_t grid_x, grid_y, grid_z;
        get_grid_coords(n.x, n.y, n.z, &grid_x, &grid_y, &grid_z);

        Supervoxel* sv = get_supervoxel(grid, grid_x, grid_y, grid_z);
        if (!sv) continue;

        float img_val = (float)img[i];
        sv->c += img_val;
        sv->x += n.x;
        sv->y += n.y;
        sv->z += n.z;
        sv->n++;

        float ksize = (float)sv->n;
        float c_over_ksize = sv->c/ksize;
        float x_over_ksize = sv->x/ksize;
        float y_over_ksize = sv->y/ksize;
        float z_over_ksize = sv->z/ksize;

        static const int8_t offsets[6][3] = {
            {0, 0, 1}, {0, 0, -1}, {0, 1, 0},
            {0, -1, 0}, {1, 0, 0}, {-1, 0, 0}
        };

        for(int neighbor = 0; neighbor < 6; neighbor++) {
            int16_t zz = n.z + offsets[neighbor][2];
            int16_t yy = n.y + offsets[neighbor][1];
            int16_t xx = n.x + offsets[neighbor][0];

            if (zz < 0 || zz >= VOLUME_DIM ||
                yy < 0 || yy >= VOLUME_DIM ||
                xx < 0 || xx >= VOLUME_DIM)
                continue;

            int ii = (zz * VOLUME_SLICE) + (xx * VOLUME_DIM) + yy;
            if (visited[ii] || img[ii] == 0) continue;  // Skip visited and zero voxels

            float dc = scale * (c_over_ksize - (float)img[ii]);
            float dx = x_over_ksize - xx;
            float dy = y_over_ksize - yy;
            float dz = z_over_ksize - zz;

            float intensity_factor = (float)img[ii] / 255.0f;
            float spatial_distance = dx*dx + dy*dy + dz*dz;
            float d = (dc*dc + spatial_distance*invwt*(1.0f + intensity_factor*intensity_weight)) / ksize;

            heap_push(&pq, (HeapNode){
                .d = d,
                .x = (uint8_t)xx,
                .y = (uint8_t)yy,
                .z = (uint8_t)zz
            });
        }
    }

    // Final averaging
    for (int grid_z = 0; grid_z < GRID_DIM; grid_z++) {
        for (int grid_y = 0; grid_y < GRID_DIM; grid_y++) {
            for (int grid_x = 0; grid_x < GRID_DIM; grid_x++) {
                Supervoxel* sv = get_supervoxel(grid, grid_x, grid_y, grid_z);
                if (sv && sv->n > 0) {
                    sv->c /= sv->n;
                    sv->x /= sv->n;
                    sv->y /= sv->n;
                    sv->z /= sv->n;
                }
            }
        }
    }

    free(visited);
    heap_free(&pq);
    return GRID_DIM * GRID_DIM * GRID_DIM;  // Return total number of supervoxels
}

#ifdef __cplusplus
}
#endif