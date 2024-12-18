
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>

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

  // Initialize all labels to 0 (null superpixel)
  for (int z = 0; z < lz; z++) {
    for (int y = 0; y < ly; y++) {
      for (int x = 0; x < lx; x++) {
        labels[idx(z,y,x)] = 0;
      }
    }
  }

  // Create seeds at regular intervals regardless of value
  for (u16 iz = 0; iz < lz; iz += d_seed) {
    for (u16 iy = 0; iy < ly; iy += d_seed) {
      for (u16 ix = 0; ix < lx; ix += d_seed) {
        numk++;
        if (img[idx(iz,iy,ix)] > 0) {
          // If seed location is non-zero, use it directly
          heap_push(&pq, (HeapNode){.d = 0.0f, .k = numk, .x = ix, .y = iy, .z = iz});
        } else {
          // If seed location is zero, look for nearest non-zero voxel in neighborhood
          int found = 0;
          for (int r = 1; r <= d_seed/2 && !found; r++) {  // Search up to half the seed spacing
            for (int dz = -r; dz <= r && !found; dz++) {
              for (int dy = -r; dy <= r && !found; dy++) {
                for (int dx = -r; dx <= r && !found; dx++) {
                  if (dx*dx + dy*dy + dz*dz > r*r) continue;  // Skip corners of cube

                  int xx = ix + dx;
                  int yy = iy + dy;
                  int zz = iz + dz;

                  if (xx < 0 || xx >= lx || yy < 0 || yy >= ly || zz < 0 || zz >= lz)
                    continue;

                  if (img[idx(zz,yy,xx)] > 0) {
                    heap_push(&pq, (HeapNode){.d = 0.0f, .k = numk, .x = xx, .y = yy, .z = zz});
                    found = 1;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (numk == 0) {
    return 0;
  }

  f32 invwt = compactness * compactness * (f32)numk / (f32)img_size;
  NeighborNode neighbors[26];

  while (pq.len > 0) {
    HeapNode n = heap_pop(&pq);
    int i = idx(n.z, n.y, n.x);

    if (labels[i] > 0) continue;

    // Only grow into non-zero voxels
    if (img[i] == 0) continue;

    u32 k = n.k;
    labels[i] = k;
    superpixels[k].c += (f32)img[i];
    superpixels[k].x += (f32)n.x;
    superpixels[k].y += (f32)n.y;
    superpixels[k].z += (f32)n.z;
    superpixels[k].n += 1;

    // Find all valid neighbors
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

          // Only consider unlabeled, non-zero voxels as valid neighbors
          if (labels[ni] <= 0 && img[ni] > 0) {
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

    // Sort and push neighbors
    for (int i = 0; i < neighbor_count - 1 && i < MAX_NEIGHBORS; i++) {
      for (int j = 0; j < neighbor_count - i - 1; j++) {
        if (neighbors[j].d > neighbors[j + 1].d) {
          NeighborNode temp = neighbors[j];
          neighbors[j] = neighbors[j + 1];
          neighbors[j + 1] = temp;
        }
      }
    }

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
    if (ksize > 0) {  // Only normalize if superpixel has any voxels
      superpixels[k].c /= ksize;
      superpixels[k].x /= ksize;
      superpixels[k].y /= ksize;
      superpixels[k].z /= ksize;
    }
  }

  heap_free(&pq);
  return 0;
}

typedef struct SuperpixelConnection {
    u32 neighbor_label;  // The label of the neighboring superpixel
    f32 connection_strength;  // Sum of values at the boundary
} SuperpixelConnection;

typedef struct SuperpixelConnections {
    SuperpixelConnection* connections;  // Array of connections for this superpixel
    int num_connections;  // Number of connections found
} SuperpixelConnections;

void free_superpixel_connections(SuperpixelConnections* connections, u32 num_superpixels) {
    if (!connections) return;
    for (u32 i = 1; i <= num_superpixels; i++) {
        if (connections[i].connections) {
            free(connections[i].connections);
        }
    }
    free(connections);
}

SuperpixelConnections* calculate_superpixel_connections(
    const u8* img,
    const u32* labels,
    u16 lz, u16 ly, u16 lx,
    u32 num_superpixels
) {
    int lylx = ly * lx;
    SuperpixelConnections* all_connections = (SuperpixelConnections*)calloc(
        num_superpixels + 1, sizeof(SuperpixelConnections));

    // First pass: count unique neighbors
    for (int z = 0; z < lz; z++) {
        for (int y = 0; y < ly; y++) {
            for (int x = 0; x < lx; x++) {
                u32 current_label = labels[idx(z,y,x)];
                if (current_label == 0) continue;

                for (int dz = -1; dz <= 1; dz++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dz == 0 && dy == 0 && dx == 0) continue;

                            int xx = x + dx;
                            int yy = y + dy;
                            int zz = z + dz;

                            if (xx < 0 || xx >= lx || yy < 0 || yy >= ly || zz < 0 || zz >= lz)
                                continue;

                            u32 neighbor_label = labels[idx(zz,yy,xx)];
                            if (neighbor_label == 0 || neighbor_label == current_label)
                                continue;

                            bool found = false;
                            for (int i = 0; i < all_connections[current_label].num_connections; i++) {
                                if (all_connections[current_label].connections &&
                                    all_connections[current_label].connections[i].neighbor_label == neighbor_label) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                all_connections[current_label].num_connections++;
                            }
                        }
                    }
                }
            }
        }
    }

    // Allocate connection arrays
    for (u32 i = 1; i <= num_superpixels; i++) {
        if (all_connections[i].num_connections > 0) {
            all_connections[i].connections = (SuperpixelConnection*)calloc(
                all_connections[i].num_connections, sizeof(SuperpixelConnection));

            // Initialize connections
            for (int j = 0; j < all_connections[i].num_connections; j++) {
                all_connections[i].connections[j].connection_strength = 0.0f;
            }
            all_connections[i].num_connections = 0;  // Reset for second pass
        }
    }

    // Second pass: calculate connections with running totals
    for (int z = 0; z < lz; z++) {
        for (int y = 0; y < ly; y++) {
            for (int x = 0; x < lx; x++) {
                u32 current_label = labels[idx(z,y,x)];
                if (current_label == 0) continue;
                float current_val = (float)img[idx(z,y,x)];

                for (int dz = -1; dz <= 1; dz++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dz == 0 && dy == 0 && dx == 0) continue;

                            int xx = x + dx;
                            int yy = y + dy;
                            int zz = z + dz;

                            if (xx < 0 || xx >= lx || yy < 0 || yy >= ly || zz < 0 || zz >= lz)
                                continue;

                            u32 neighbor_label = labels[idx(zz,yy,xx)];
                            if (neighbor_label == 0 || neighbor_label == current_label)
                                continue;

                            float neighbor_val = (float)img[idx(zz,yy,xx)];
                            float value_similarity = 1.0f - fabsf(current_val - neighbor_val) / 255.0f;

                            // Find or create connection entry
                            int conn_idx = -1;
                            for (int i = 0; i < all_connections[current_label].num_connections; i++) {
                                if (all_connections[current_label].connections[i].neighbor_label == neighbor_label) {
                                    conn_idx = i;
                                    break;
                                }
                            }

                            if (conn_idx == -1) {
                                conn_idx = all_connections[current_label].num_connections++;
                                all_connections[current_label].connections[conn_idx].neighbor_label = neighbor_label;
                            }

                            all_connections[current_label].connections[conn_idx].connection_strength += value_similarity;
                        }
                    }
                }
            }
        }
    }

    return all_connections;
}