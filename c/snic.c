
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "common.h"
#include "snic.h"

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
  return (Heap){.len = 0, .size = size, .nodes = (HeapNode*)calloc(size*2, sizeof(HeapNode))};
}

static void heap_free(Heap *heap) {
  free(heap->nodes);
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


static int superpixel_add_neighbors(Superpixel *superpixels, u32 k1, u32 k2) {
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

static inline void process_neighbor(
    int ndz, int ndy, int ndx, int ioffset,
    HeapNode n, int i,
    const u8* img, u32* labels, Superpixel* superpixels,
    u32 k, f32 invwt, Heap* pq, int* neigh_overflow, u32 lz, u32 ly, u32 lx
) {
    int xx = n.x + ndx;
    int yy = n.y + ndy;
    int zz = n.z + ndz;

#define sqr(x) ((x)*(x))

    if (0 <= xx && xx < lz && 0 <= yy && yy < ly && 0 <= zz && zz < lx) {
        int ii = i + ioffset;
        if (labels[ii] <= 0) {
            f32 ksize = (f32)superpixels[k].n;
            f32 dc = sqr(255.0f*(superpixels[k].c - (img[ii]*ksize)));
            f32 dx = superpixels[k].x - xx*ksize;
            f32 dy = superpixels[k].y - yy*ksize;
            f32 dz = superpixels[k].z - zz*ksize;
            f32 dpos = sqr(dx) + sqr(dy) + sqr(dz);
            f32 d = (dc + dpos*invwt) / (ksize*ksize);
            heap_push(pq, (HeapNode){.d = d, .k = k, .x = (u16)xx, .y = (u16)yy, .z = (u16)zz});
        } else if (k != labels[ii]) {
            *neigh_overflow += superpixel_add_neighbors(superpixels, k, labels[ii]);
            *neigh_overflow += superpixel_add_neighbors(superpixels, labels[ii], k);
        }
    }
}

int snic(const u8* img, u16 lz, u16 ly, u16 lx, u32 d_seed, f32 compactness, u32* labels, Superpixel* superpixels) {
  int neigh_overflow = 0;
  int lylx = ly * lx;
  int img_size = lz * ly * lx;
  #define idx(z, y, x) ((z)*lylx + (x)*lx + (y))
  #define sqr(x) ((x)*(x))

  // Structure to store neighbor information temporarily
  typedef struct {
    float dist;
    int dz, dy, dx;
    int offset;
  } NeighborInfo;

  Heap pq = heap_alloc(img_size);
  u32 numk = 0;
  for (u16 iz = 0; iz < lz; iz += d_seed) {
    for (u16 iy = 0; iy < ly; iy += d_seed) {
      for (u16 ix = 0; ix < lx; ix += d_seed) {
        numk++;
        u16 x = ix, y = iy, z = iz;
        heap_push(&pq, (HeapNode){.d = 0.0f, .k = numk, .x = x, .y = y, .z = z});
      }
    }
  }
  if (numk == 0) {
    return 0;
  }

  f32 invwt = compactness * compactness * (f32)numk / (f32)img_size;

  // Array to store neighbor information
  NeighborInfo neighbors[26];

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

    // First, check all 26 neighbors and store their information
    int neighbor_count = 0;
    for (int dz = -1; dz <= 1; dz++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          // Skip the center point
          if (dz == 0 && dy == 0 && dx == 0) continue;

          // Check if neighbor is within bounds
          if (n.z + dz < 0 || n.z + dz >= lz ||
              n.y + dy < 0 || n.y + dy >= ly ||
              n.x + dx < 0 || n.x + dx >= lx) continue;

          int offset = dz * lylx + dx * lx + dy;
          int ni = i + offset;

          // Skip if already labeled
          if (labels[ni] > 0) continue;

          // Calculate distance metric (same as in process_neighbor)
          f32 dc = (f32)img[ni] - (f32)img[i];
          f32 dx_dist = (f32)dx;
          f32 dy_dist = (f32)dy;
          f32 dz_dist = (f32)dz;
          f32 ds = sqr(dx_dist) + sqr(dy_dist) + sqr(dz_dist);
          f32 dist = sqr(dc) + ds * invwt;

          neighbors[neighbor_count] = (NeighborInfo){
            .dist = dist,
            .dz = dz,
            .dy = dy,
            .dx = dx,
            .offset = offset
          };
          neighbor_count++;
        }
      }
    }

    // Sort neighbors by distance (simple bubble sort since we only need top 6)
    for (int i = 0; i < 6 && i < neighbor_count; i++) {
      for (int j = i + 1; j < neighbor_count; j++) {
        if (neighbors[j].dist < neighbors[i].dist) {
          NeighborInfo temp = neighbors[i];
          neighbors[i] = neighbors[j];
          neighbors[j] = temp;
        }
      }
    }

    // Process only the closest 6 neighbors (or fewer if there aren't 6 valid neighbors)
    int num_to_process = (neighbor_count < 6) ? neighbor_count : 6;
    for (int j = 0; j < num_to_process; j++) {
      NeighborInfo* neigh = &neighbors[j];
      process_neighbor(neigh->dz, neigh->dy, neigh->dx, neigh->offset,
                      n, i, img, labels, superpixels, k, invwt,
                      &pq, &neigh_overflow, lz, ly, lx);
    }
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
