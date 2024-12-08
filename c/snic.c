#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#define D_SEED 8
#define DIMENSION 256
#define COMPACTNESS 1000.0f

typedef float f32;
typedef unsigned int u32;
typedef unsigned short u16;

typedef struct HeapNode {
    f32 d;      // distance/priority
    u32 k;      // superpixel index
    u16 x, y, z; // coordinates
} HeapNode;

typedef struct Heap {
    int len;
    int capacity;
    HeapNode* nodes;
} Heap;

// Optimized initial heap size calculation
static inline int calculate_initial_heap_size() {
    // Instead of allocating for full image size * 8, we calculate based on active frontier
    // Maximum frontier size is approximately 6 * current_layer_size
    // For 3D grid with step D_SEED, calculate max active frontier
    int layer_size = (DIMENSION / D_SEED) * (DIMENSION / D_SEED);
    int max_frontier = 6 * layer_size; // 6 faces of expansion

    // Add 20% buffer for safety
    return (int)(max_frontier * 1.2);
}

static inline size_t round_to_page_size(size_t size) {
    long page_size = sysconf(_SC_PAGESIZE);
    return (size + page_size - 1) & ~(page_size - 1);
}

static inline Heap heap_alloc(int suggested_capacity) {
    int initial_capacity = calculate_initial_heap_size();
    // Use the larger of calculated size or suggested capacity
    initial_capacity = initial_capacity > suggested_capacity ? initial_capacity : suggested_capacity;

    // Round to power of 2 for efficient resizing
    initial_capacity = 1 << (32 - __builtin_clz(initial_capacity - 1));

    size_t alloc_size = round_to_page_size(initial_capacity * sizeof(HeapNode));
    printf("Initial heap allocation: %zu bytes (capacity: %d nodes)\n",
           alloc_size, initial_capacity);

    HeapNode* nodes = (HeapNode*)mmap(
        NULL,
        alloc_size,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );

    if (nodes == MAP_FAILED) {
        perror("mmap failed");
        return (Heap){0};
    }

    nodes[0] = (HeapNode){0};
    return (Heap){.len = 0, .capacity = initial_capacity, .nodes = nodes};
}

// Rest of heap implementation remains similar but with more aggressive shrinking
static inline int heap_resize(Heap *heap, int new_capacity) {
    // More aggressive shrinking threshold
    if (new_capacity < heap->capacity && heap->len < heap->capacity / 8) {
        new_capacity = heap->capacity / 2;
    }

    new_capacity = 1 << (32 - __builtin_clz(new_capacity - 1));
    size_t new_size = round_to_page_size(new_capacity * sizeof(HeapNode));

    HeapNode* new_nodes = (HeapNode*)mmap(
        NULL,
        new_size,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );

    if (new_nodes == MAP_FAILED) {
        perror("mmap failed during resize");
        return 0;
    }

    memcpy(new_nodes, heap->nodes, (heap->len + 1) * sizeof(HeapNode));

    size_t old_size = round_to_page_size(heap->capacity * sizeof(HeapNode));
    if (munmap(heap->nodes, old_size) == -1) {
        perror("munmap failed during resize");
        munmap(new_nodes, new_size);
        return 0;
    }

    heap->nodes = new_nodes;
    heap->capacity = new_capacity;
    return 1;
}

// Optimized heap operations
#define heap_parent(i) ((i) >> 1)
#define heap_left(i) ((i) << 1)
#define heap_right(i) (((i) << 1) | 1)
#define heap_node_val(n) (-n.d)

static inline void heap_push(Heap *heap, HeapNode node) {
    if (heap->len + 1 >= heap->capacity) {
        if (!heap_resize(heap, heap->capacity * 2)) {
            printf("Failed to grow heap\n");
            return;
        }
    }

    int i = ++heap->len;
    while (i > 1) {
        int p = heap_parent(i);
        if (heap_node_val(heap->nodes[p]) >= heap_node_val(node)) break;
        heap->nodes[i] = heap->nodes[p];
        i = p;
    }
    heap->nodes[i] = node;
}

static inline HeapNode heap_pop(Heap *heap) {
    if (heap->len == 0) return (HeapNode){0};

    HeapNode result = heap->nodes[1];
    HeapNode last = heap->nodes[heap->len--];

    // More aggressive shrinking
    if (heap->len > 0 && heap->len < heap->capacity / 8) {
        heap_resize(heap, heap->capacity / 2);
    }

    int i = 1;
    while (1) {
        int largest = i;
        int l = heap_left(i);
        int r = heap_right(i);

        if (l <= heap->len && heap_node_val(heap->nodes[l]) > heap_node_val(last)) largest = l;
        if (r <= heap->len && heap_node_val(heap->nodes[r]) > heap_node_val(heap->nodes[largest])) largest = r;

        if (largest == i) break;

        heap->nodes[i] = heap->nodes[largest];
        i = largest;
    }
    heap->nodes[i] = last;

    return result;
}

static inline void heap_free(Heap *heap) {
    if (heap->nodes) {
        size_t alloc_size = round_to_page_size((size_t)heap->capacity * sizeof(HeapNode));
        if (munmap(heap->nodes, alloc_size) == -1) {
            perror("munmap failed");
        }
        heap->nodes = NULL;
        heap->len = 0;
        heap->capacity = 0;
    }
}

// SNIC ////////////////////////////////////////////////////////////////////////

// This is based on the paper and the code from:
// - https://www.epfl.ch/labs/ivrl/research/snic-superpixels/
// - https://github.com/achanta/SNIC/

// There isn't a theoretical maximum for SNIC neighbors. The neighbors of a cube
// would be 26, so if compactness is high we shouldn't exceed that by too much.
// 56 results in sizeof(Superpixel) == 4*8*8 (4 64B cachelines).
#define SUPERPIXEL_MAX_NEIGHS (64)
typedef struct Superpixel {
  f32 x, y, z, c;
  u32 n;
  u32 neighs[SUPERPIXEL_MAX_NEIGHS];
} Superpixel;

int snic_superpixel_max_neighs()  {
  return SUPERPIXEL_MAX_NEIGHS;
}

int superpixel_add_neighbors(Superpixel *superpixels, u32 k1, u32 k2) {
  int i = 0;
  for (; i < SUPERPIXEL_MAX_NEIGHS; i++) {
    if (superpixels[k1].neighs[i] == 0) {
      superpixels[k1].neighs[i] = k2;
      return 0;
    } else if (superpixels[k1].neighs[i] == k2) {
      return 0;
    }
  }
  return 1;
}

int snic_superpixel_count()  {
  int cz = (DIMENSION - D_SEED/2 + D_SEED - 1)/D_SEED;
  int cy = (DIMENSION - D_SEED/2 + D_SEED - 1)/D_SEED;
  int cx = (DIMENSION - D_SEED/2 + D_SEED - 1)/D_SEED;
  return cx*cy*cz;
}

// Modified SNIC implementation with optimized neighbor selection
int snic(f32 *img, u32 *labels, Superpixel* superpixels) {
    int neigh_overflow = 0;
    int lylx = DIMENSION * DIMENSION;
    int img_size = lylx * DIMENSION;
    #define idx(z, y, x) ((z)*lylx + (x)*DIMENSION + (y))
    #define sqr(x) ((x)*(x))

    // Initialize priority queue with seeds on a grid
    Heap pq = heap_alloc(256*256*256); // Start small, will grow as needed
    u32 numk = 0;
    for (u16 iz = 0; iz < DIMENSION; iz += D_SEED) {
        for (u16 iy = 0; iy < DIMENSION; iy += D_SEED) {
            for (u16 ix = 0; ix < DIMENSION; ix += D_SEED) {
                numk++;
                heap_push(&pq, (HeapNode){.d = 0.0f, .k = numk, .x = ix, .y = iy, .z = iz});
            }
        }
    }
    printf("placed %d superpixel seeds\n", numk);

    f32 invwt = (COMPACTNESS*COMPACTNESS*numk)/(f32)(img_size);

    // Temporary storage for neighbor evaluation
    typedef struct {
        f32 d;
        u16 x, y, z;
    } NeighborInfo;
    NeighborInfo neighbors[6]; // For 6-connectivity

    while (pq.len > 0) {
        HeapNode n = heap_pop(&pq);
        int i = idx(n.z, n.y, n.x);
        if (labels[i] > 0) continue;

        u32 k = n.k;
        labels[i] = k;
        superpixels[k].c += img[i];
        superpixels[k].x += n.x;
        superpixels[k].y += n.y;
        superpixels[k].z += n.z;
        superpixels[k].n += 1;

        // Evaluate all neighbors first
        int num_neighbors = 0;
        #define eval_neigh(ndz, ndy, ndx, ioffset) { \
            int xx = n.x + ndx; int yy = n.y + ndy; int zz = n.z + ndz; \
            if (0 <= xx && xx < DIMENSION && 0 <= yy && yy < DIMENSION && 0 <= zz && zz < DIMENSION) { \
                int ii = i + ioffset; \
                if (labels[ii] <= 0) { \
                    f32 ksize = (f32)superpixels[k].n; \
                    f32 dc = sqr(100.0f*(superpixels[k].c - (img[ii]*ksize))); \
                    f32 dx = superpixels[k].x - xx*ksize; \
                    f32 dy = superpixels[k].y - yy*ksize; \
                    f32 dz = superpixels[k].z - zz*ksize; \
                    f32 dpos = sqr(dx) + sqr(dy) + sqr(dz); \
                    f32 d = (dc + dpos*invwt) / (ksize*ksize); \
                    neighbors[num_neighbors++] = (NeighborInfo){.d = d, .x = xx, .y = yy, .z = zz}; \
                } else if (k != labels[ii]) { \
                    neigh_overflow += superpixel_add_neighbors(superpixels, k, labels[ii]); \
                    neigh_overflow += superpixel_add_neighbors(superpixels, labels[ii], k); \
                } \
            } \
        }

        eval_neigh( 0,  1,  0,    1);
        eval_neigh( 0, -1,  0,   -1);
        eval_neigh( 0,  0,  1,    DIMENSION);
        eval_neigh( 0,  0, -1,   -DIMENSION);
        eval_neigh( 1,  0,  0,    lylx);
        eval_neigh(-1,  0,  0,   -lylx);

        // Only add the best N neighbors to the heap
        // Here we're selecting the 2 best neighbors (can be tuned)
        #define BEST_NEIGHBORS  2
        for (int j = 0; j < num_neighbors && j < BEST_NEIGHBORS; j++) {
            // Find best remaining neighbor
            int best_idx = j;
            f32 best_d = neighbors[j].d;
            for (int m = j + 1; m < num_neighbors; m++) {
                if (neighbors[m].d < best_d) {
                    best_d = neighbors[m].d;
                    best_idx = m;
                }
            }
            // Swap if needed
            if (best_idx != j) {
                NeighborInfo tmp = neighbors[j];
                neighbors[j] = neighbors[best_idx];
                neighbors[best_idx] = tmp;
            }
            // Add to heap
            heap_push(&pq, (HeapNode){
                .d = neighbors[j].d,
                .k = k,
                .x = neighbors[j].x,
                .y = neighbors[j].y,
                .z = neighbors[j].z
            });
        }
    }

    // Finalize superpixel centers
    for (u32 k = 1; k <= numk; k++) {
        f32 ksize = (f32)superpixels[k].n;
        superpixels[k].c /= ksize;
        superpixels[k].x /= ksize;
        superpixels[k].y /= ksize;
        superpixels[k].z /= ksize;
    }

    #undef sqr
    #undef idx
    heap_free(&pq);
    return neigh_overflow;
}