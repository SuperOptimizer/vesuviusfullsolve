
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "common.h"
#include "snic.h"

typedef struct NeighborNode {
  f32 d;
  u16 x, y, z;
} NeighborNode;

#define idx(z, y, x) ((z)*lylx + (x)*lx + (y))
#define sqr(x) ((x)*(x))

typedef struct HeapNode {
  f32 d;
  u32 k;
  u16 z, y, x;
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

static Heap heap_alloc(int size) {
  size_t total_size = size * 2 * 2 * 2 * sizeof(HeapNode);
  void* mem = mmap(NULL, total_size,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);

  if (mem == MAP_FAILED) {
    // Handle allocation failure
    return (Heap){.len = 0, .size = 0, .nodes = NULL};
  }

  return (Heap){
    .len = 0,
    .size = size,
    .nodes = (HeapNode*)mem
  };
}

static void heap_free(Heap *heap) {
  if (heap->nodes) {
    size_t total_size = heap->size * 2 * 2 * 2 * sizeof(HeapNode);
    munmap(heap->nodes, total_size);
    heap->nodes = NULL;
  }
}

static void heap_push(Heap *heap, HeapNode node) {
  heap->len++;
  heap->nodes[heap->len] = node;
  for (int i = heap->len, j = 0; i > 1; i = j) {
    j = heap_parent(i);
    heap_fix_edge(heap, j, i) else break;
  }
}

static HeapNode heap_pop(Heap *heap) {
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



int snic(const u8* img, u16 lz, u16 ly, u16 lx, u32 d_seed, f32 compactness, u32* labels, Superpixel* superpixels) {
  int lylx = ly * lx;
  int img_size = lz * ly * lx;
  Heap pq = heap_alloc(img_size);
  u32 numk = 0;

for (int z = 0; z < lz; z++) {
  for (int y = 0; y < ly; y++) {
    for (int x = 0; x < lx; x++) {
      labels[idx(z,y,x)] = 0;
    }
  }
}

  // Initial seeds remain the same
  for (u16 iz = 0; iz < lz; iz += d_seed) {
    for (u16 iy = 0; iy < ly; iy += d_seed) {
      for (u16 ix = 0; ix < lx; ix += d_seed) {
        numk++;
        heap_push(&pq, (HeapNode){.d = 0.0f, .k = numk, .x = ix, .y = iy, .z = iz});
      }
    }
  }
  if (numk == 0) {
    return 0;
  }

  f32 invwt = compactness * compactness * (f32)numk / (f32)img_size;

  // Temporary array to store and sort neighbors
  NeighborNode neighbors[26];  // Still need space for all neighbors to find best N

  while (pq.len > 0) {
    HeapNode n = heap_pop(&pq);
    int i = idx(n.z, n.y, n.x);

    if (labels[i] > 0) continue;

    u32 k = n.k;
    labels[i] = k;
    superpixels[k].c += (f32)img[i];
    superpixels[k].x += (f32)n.x;
    superpixels[k].y += (f32)n.y;
    superpixels[k].z += (f32)n.z;
    superpixels[k].n += 1;

    // Find all valid neighbors first
    int neighbor_count = 0;
    for (int dz = -1; dz <= 1; dz++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          if (dz == 0 && dy == 0 && dx == 0) continue;

          int xx = n.x + dx;
          int yy = n.y + dy;
          int zz = n.z + dz;

          if (xx < 0 || xx >= lx || yy < 0 || yy >= ly || zz < 0 || zz >= lz)
            continue;

          int offset = dz * lylx + dx * lx + dy;
          int ni = i + offset;

          if (labels[ni] <= 0) {
            f32 ksize = (f32)superpixels[k].n;
            f32 dc = sqr(255.0f*(superpixels[k].c - (img[ni]*ksize)));
            f32 dx_pos = superpixels[k].x - xx*ksize;
            f32 dy_pos = superpixels[k].y - yy*ksize;
            f32 dz_pos = superpixels[k].z - zz*ksize;
            f32 dpos = sqr(dx_pos) + sqr(dy_pos) + sqr(dz_pos);
            f32 d = (dc + dpos*invwt) / (ksize*ksize);

            neighbors[neighbor_count] = (NeighborNode){
              .d = d,
              .x = (u16)xx,
              .y = (u16)yy,
              .z = (u16)zz
            };
            neighbor_count++;
          }
        }
      }
    }

    // Simple bubble sort to find N best neighbors
    // For small N, bubble sort is fine. For larger N, consider quickselect
    for (int i = 0; i < neighbor_count - 1 && i < MAX_NEIGHBORS; i++) {
      for (int j = 0; j < neighbor_count - i - 1; j++) {
        if (neighbors[j].d > neighbors[j + 1].d) {
          NeighborNode temp = neighbors[j];
          neighbors[j] = neighbors[j + 1];
          neighbors[j + 1] = temp;
        }
      }
    }

    // Push only the N best neighbors
    for (int i = 0; i < neighbor_count && i < MAX_NEIGHBORS; i++) {
      heap_push(&pq, (HeapNode){
        .d = neighbors[i].d,
        .k = k,
        .x = neighbors[i].x,
        .y = neighbors[i].y,
        .z = neighbors[i].z
      });
    }
  }

  // Normalize superpixel properties
  for (u32 k = 1; k <= numk; k++) {
    f32 ksize = (f32)superpixels[k].n;
    superpixels[k].c /= ksize;
    superpixels[k].x /= ksize;
    superpixels[k].y /= ksize;
    superpixels[k].z /= ksize;
  }

  heap_free(&pq);
  return 0;
}