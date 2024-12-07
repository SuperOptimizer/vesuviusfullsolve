#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// Constants for 256³ volume
#define D_SEED 2
#define VOLUME_DIM 256
#define VOLUME_SLICE (VOLUME_DIM * VOLUME_DIM)
#define ZERO_SUPERVOXEL_INDEX 0  // Reserved index for zero-valued voxels

typedef struct HeapNode {
    unsigned int k;
    uint8_t x, y, z;
    uint8_t d;
} HeapNode;

#define heap_node_val(n) (-n.d)

typedef struct Heap {
    int len, size;
    HeapNode* nodes;
} Heap;

typedef struct Supervoxel {
    float x, y, z, c;  // Keep as float for accumulation and averaging
    unsigned int n;
} Supervoxel;

static inline Heap heap_alloc(int size) {
    size_t alloc_size = (size_t)size * sizeof(HeapNode);
    printf("mmap allocating %llu bytes\n", (unsigned long long)alloc_size);

    HeapNode* nodes = (HeapNode*)mmap(
        NULL,                   // Let OS choose address
        alloc_size,            // Size in bytes
        PROT_READ | PROT_WRITE, // Read and write permissions
        MAP_PRIVATE | MAP_ANONYMOUS, // Private pages, not backed by file
        -1,                    // No file descriptor needed
        0                      // No offset
    );

    if (nodes == MAP_FAILED) {
        perror("mmap failed");
        return (Heap){0};
    }

    // Initialize first node to zero
    nodes[0] = (HeapNode){0};

    return (Heap){.len = 0, .size = size, .nodes = nodes};
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
        if(heap_node_val(nodes[parent]) >= heap_node_val(node)) break;
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
        if(child < heap->len && heap_node_val(nodes[child]) < heap_node_val(nodes[child + 1]))
            child++;
        if(heap_node_val(last) >= heap_node_val(nodes[child])) break;
        nodes[i] = nodes[child];
        i = child;
    }
    nodes[i] = last;
    return result;
}



static inline int snic_supervoxel_count() {
    return (256/D_SEED) * (256/D_SEED) * (256/D_SEED);
}

// Pre-process to initialize zero supervoxel
static void init_zero_supervoxel(uint8_t *img, Supervoxel* supervoxels, int img_size) {
    // Initialize zero supervoxel
    supervoxels[ZERO_SUPERVOXEL_INDEX] = (Supervoxel){0};

    // Collect zero-valued voxels
    for (int i = 0; i < img_size; i++) {
        if (img[i] == 0) {
            Supervoxel* sv = &supervoxels[ZERO_SUPERVOXEL_INDEX];
            sv->c += 0;
            sv->x += (i / VOLUME_DIM) % VOLUME_DIM;
            sv->y += i % VOLUME_DIM;
            sv->z += i / VOLUME_SLICE;
            sv->n++;
        }
    }
}

// Add these new constants for adaptive seeding
#define MIN_INTENSITY 1   // Minimum intensity to consider (excluding 0)
#define MAX_INTENSITY 255 // Maximum intensity value
#define DENSITY_FACTOR 2.0f // How much to adjust density based on intensity

static inline float calculate_local_density(uint8_t* img, int x, int y, int z, int window_size) {
    float sum = 0;
    int count = 0;
    int half_window = window_size / 2;

    for(int dz = -half_window; dz <= half_window; dz++) {
        for(int dy = -half_window; dy <= half_window; dy++) {
            for(int dx = -half_window; dx <= half_window; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                if(nx >= 0 && nx < VOLUME_DIM &&
                   ny >= 0 && ny < VOLUME_DIM &&
                   nz >= 0 && nz < VOLUME_DIM) {
                    int idx = (nz * VOLUME_SLICE) + (nx * VOLUME_DIM) + ny;
                    if(img[idx] > 0) {  // Only consider non-zero values
                        sum += img[idx];
                        count++;
                    }
                }
            }
        }
    }

    return count > 0 ? sum / (float)count : 0.0f;
}

static inline int should_create_seed(uint8_t* img, int x, int y, int z, float base_spacing, float local_density) {
    // Skip if current position is a zero-valued voxel
    int idx = (z * VOLUME_SLICE) + (x * VOLUME_DIM) + y;
    if(img[idx] == 0) return 0;

    // Calculate adaptive spacing based on local density
    float intensity_factor = local_density / MAX_INTENSITY;
    float adjusted_spacing = base_spacing * (1.0f + (1.0f - intensity_factor) * DENSITY_FACTOR);

    // Check if current position aligns with adjusted grid
    float rel_x = (float)x / adjusted_spacing;
    float rel_y = (float)y / adjusted_spacing;
    float rel_z = (float)z / adjusted_spacing;

    // Create seed if position is close to adjusted grid intersection
    float threshold = 0.2f;  // Tolerance for grid alignment
    return (fabs(rel_x - round(rel_x)) < threshold &&
            fabs(rel_y - round(rel_y)) < threshold &&
            fabs(rel_z - round(rel_z)) < threshold);
}

int supervoxeler(uint8_t *img, Supervoxel* supervoxels) {
    int img_size = VOLUME_DIM * VOLUME_DIM * VOLUME_DIM;

    // Initialize zero supervoxel
    init_zero_supervoxel(img, supervoxels, img_size);

    // Base spacing for initial uniform grid
    float base_spacing = (float)D_SEED;

    Heap pq = heap_alloc(img_size*16);
    unsigned int numk = 0;

    // Create boolean array to track visited voxels
    uint8_t* visited = (uint8_t*)calloc(img_size, sizeof(uint8_t));
    if (!visited) {
        heap_free(&pq);
        return -1;
    }

    // Mark zero voxels as visited
    for (int i = 0; i < img_size; i++) {
        if (img[i] == 0) {
            visited[i] = 1;
        }
    }

    // Modified seeding loop with adaptive density
    for(int z = 0; z < VOLUME_DIM; z++) {
        for(int y = 0; y < VOLUME_DIM; y++) {
            for(int x = 0; x < VOLUME_DIM; x++) {
                int idx = (z * VOLUME_SLICE) + (x * VOLUME_DIM) + y;
                if (visited[idx]) continue;

                float local_density = calculate_local_density(img, x, y, z, D_SEED);

                if(should_create_seed(img, x, y, z, base_spacing, local_density)) {
                    heap_push(&pq, (HeapNode){
                        .d = 0,
                        .k = ++numk,
                        .x = (uint8_t)x,
                        .y = (uint8_t)y,
                        .z = (uint8_t)z
                    });
                }
            }
        }
    }

    float invwt = (numk)/(float)(img_size);
    const float scale = 100.0f;
    const float intensity_weight = 2.0f;

    while (pq.len > 0) {
        HeapNode n = heap_pop(&pq);
        int i = ((int)n.z * VOLUME_SLICE) + ((int)n.x * VOLUME_DIM) + n.y;

        if (visited[i]) continue;
        visited[i] = 1;

        unsigned int k = n.k;
        Supervoxel* sv = &supervoxels[k];
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

        for(int neighbor = 0; neighbor < 6; neighbor++) {
            static const int8_t offsets[6][3] = {
                {0, 0, 1}, {0, 0, -1}, {0, 1, 0},
                {0, -1, 0}, {1, 0, 0}, {-1, 0, 0}
            };

            int16_t zz = n.z + offsets[neighbor][2];
            int16_t yy = n.y + offsets[neighbor][1];
            int16_t xx = n.x + offsets[neighbor][0];

            if (zz < 0 || zz >= VOLUME_DIM || yy < 0 || yy >= VOLUME_DIM || xx < 0 || xx >= VOLUME_DIM)
                continue;

            int ii = (zz * VOLUME_SLICE) + (xx * VOLUME_DIM) + yy;

            if (visited[ii]) continue;

            float dc = scale * (c_over_ksize - (float)img[ii]);
            float dx = x_over_ksize - xx;
            float dy = y_over_ksize - yy;
            float dz = z_over_ksize - zz;

            float intensity_factor = (float)img[ii] / MAX_INTENSITY;
            float spatial_distance = (dx*dx + dy*dy + dz*dz);
            float d = (dc*dc + spatial_distance*invwt*(1.0f + intensity_factor*intensity_weight)) / ksize;

            heap_push(&pq, (HeapNode){
                .d = d,
                .k = k,
                .x = (uint8_t)xx,
                .y = (uint8_t)yy,
                .z = (uint8_t)zz
            });
        }
    }

    // Final averaging
    for (unsigned int k = 0; k <= numk; k++) {
        float ksize = (float)supervoxels[k].n;
        if (ksize > 0) {
            supervoxels[k].c /= ksize;
            supervoxels[k].x /= ksize;
            supervoxels[k].y /= ksize;
            supervoxels[k].z /= ksize;
        }
    }

    free(visited);
    heap_free(&pq);
    return numk;
}