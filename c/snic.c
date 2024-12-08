#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define D_SEED 2
#define DIMENSION 256
#define COMPACTNESS 10000.0f

typedef float f32;
typedef unsigned int u32;
typedef unsigned short u16;

typedef struct HeapNode {
  f32 d;
  u32 k;
  u16 x, y, z;
  u16 pad;
} HeapNode;

#define heap_node_val(n)  (-n.d)

typedef struct Heap {
  int len, size;
  HeapNode* nodes;
} Heap;

#define heap_left(i)  (2*(i))
#define heap_right(i) (2*(i)+1)
#define heap_parent(i) ((i)/2)
#define heap_fix_edge(heap, i, j) \
  if (heap_node_val(heap->nodes[j]) > heap_node_val(heap->nodes[i])) { \
    HeapNode tmp = heap->nodes[j]; \
    heap->nodes[j] = heap->nodes[i]; \
    heap->nodes[i] = tmp; \
  }


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
  //assert(heap->len <= heap->size);

  heap->len++;
  heap->nodes[heap->len] = node;
  for (int i = heap->len, j = 0; i > 1; i = j) {
    j = heap_parent(i);
    heap_fix_edge(heap, j, i) else break;
  }
}

static inline HeapNode heap_pop(Heap *heap) {
  //assert(heap->len > 0);

  HeapNode node = heap->nodes[1];
  heap->len--;
  heap->nodes[1] = heap->nodes[heap->len+1];
  for (int i = 1, j = 0; i <= heap->len; i = j) {
    int l = heap_left(i);
    int r = heap_right(i);
    if (l > heap->len) {
      break;
    }
    j = l;
    if (r <= heap->len && heap_node_val(heap->nodes[l]) < heap_node_val(heap->nodes[r])) {
      j = r;
    } else {
    }
    heap_fix_edge(heap, i, j) else break;
  }

  return node;
}

#undef heap_left
#undef heap_right
#undef heap_parent
#undef heap_fix_edge


// SNIC ////////////////////////////////////////////////////////////////////////

// This is based on the paper and the code from:
// - https://www.epfl.ch/labs/ivrl/research/snic-superpixels/
// - https://github.com/achanta/SNIC/

// There isn't a theoretical maximum for SNIC neighbors. The neighbors of a cube
// would be 26, so if compactness is high we shouldn't exceed that by too much.
// 56 results in sizeof(Superpixel) == 4*8*8 (4 64B cachelines).
#define SUPERPIXEL_MAX_NEIGHS (256)
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

// The labels must be the same size as img, and all zeros.
int snic(f32 *img, u32 *labels, Superpixel* superpixels) {
  int neigh_overflow = 0; // Number of neighbors that couldn't be added.
  int lylx = DIMENSION * DIMENSION;
  int img_size = lylx * DIMENSION;
  #define idx(z, y, x) ((z)*lylx + (x)*DIMENSION + (y))
  #define sqr(x) ((x)*(x))

  // Initialize priority queue with seeds on a grid with step D_SEED.
  Heap pq = heap_alloc(img_size*8);
  u32 numk = 0;
  for (u16 iz = 0; iz < DIMENSION; iz += D_SEED) {
  for (u16 iy = 0; iy < DIMENSION; iy += D_SEED) {
    for (u16 ix = 0; ix < DIMENSION; ix += D_SEED) {
      numk++;
      heap_push(&pq, (HeapNode){.d = 0.0f, .k = numk, .x = ix, .y = iy, .z = iz});
    }
  }
}
printf("placed %d superpixel seeds\n",numk);
  if (numk == 0) {
    return 0;
  }

  f32 invwt = (COMPACTNESS*COMPACTNESS*numk)/(f32)(img_size);

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

    #define do_neigh(ndz, ndy, ndx, ioffset) { \
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
          heap_push(&pq, (HeapNode){.d = d, .k = k, .x = (u16)xx, .y = (u16)yy, .z = (u16)zz}); \
        } else if (k != labels[ii]) { \
          neigh_overflow += superpixel_add_neighbors(superpixels, k, labels[ii]); \
          neigh_overflow += superpixel_add_neighbors(superpixels, labels[ii], k); \
        } \
      } \
    }

    do_neigh( 0,  1,  0,    1);
    do_neigh( 0, -1,  0,   -1);
    do_neigh( 0,  0,  1,    DIMENSION);
    do_neigh( 0,  0, -1,   -DIMENSION);
    do_neigh( 1,  0,  0,  lylx);
    do_neigh(-1,  0,  0, -lylx);
    #undef do_neigh
  }

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

