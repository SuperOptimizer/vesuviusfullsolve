#pragma once

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

typedef struct HeapNode {
  __fp16 d;
  unsigned int k;
  unsigned char x, y, z;
} HeapNode;

#define heap_node_val(n) (-n.d)

typedef struct Heap {
  int len, size;
  HeapNode* nodes;
} Heap;

static inline Heap heap_alloc(int size) {
  return (Heap){.len = 0, .size = size, .nodes = (HeapNode*)calloc(size*2+1, sizeof(HeapNode))};
}

static inline void heap_free(Heap *heap) {
  free(heap->nodes);
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

typedef struct Superpixel {
  float x, y, z, c;
  unsigned int n;
} Superpixel;

static inline int snic_superpixel_count(int lx, int ly, int lz, int d_seed) {
  return (lx/d_seed) * (ly/d_seed) * (lz/d_seed);
}

void snic(float *img, int lz, int ly, int lx, int d_seed, float compactness, float lowmid, float midhig, unsigned int *labels, Superpixel* superpixels) {
  int lylx = ly * lx;
  int img_size = lylx * lz;

  // Pre-calculate constants
  float z_spacing = (float)lz / (float)(lz/d_seed);
  float y_spacing = (float)ly / (float)(ly/d_seed);
  float x_spacing = (float)lx / (float)(lx/d_seed);

  Heap pq = heap_alloc(img_size*16);
  unsigned int numk = 0;

  // Optimize seeding loop
  for (float iz = z_spacing/2; iz < lz; iz += z_spacing) {
    unsigned short z = (unsigned short)iz;
    for (float iy = y_spacing/2; iy < ly; iy += y_spacing) {
      unsigned short y = (unsigned short)iy;
      for (float ix = x_spacing/2; ix < lx; ix += x_spacing) {
        heap_push(&pq, (HeapNode){
          .d = 0.0f,
          .k = ++numk,
          .x = (unsigned short)ix,
          .y = y,
          .z = z
        });
      }
    }
  }

  float invwt = (compactness*compactness*numk)/(float)(img_size);
  const float scale = 100.0f;

  while (pq.len > 0) {
    HeapNode n = heap_pop(&pq);
    int i = n.z*lylx + n.x*ly + n.y;
    if (labels[i] > 0) continue;

    unsigned int k = n.k;
    labels[i] = k;
    Superpixel* sp = &superpixels[k];
    float img_val = img[i];
    sp->c += img_val;
    sp->x += n.x;
    sp->y += n.y;
    sp->z += n.z;
    sp->n++;

    float ksize = (float)sp->n;
    float c_over_ksize = sp->c/ksize;
    float x_over_ksize = sp->x/ksize;
    float y_over_ksize = sp->y/ksize;
    float z_over_ksize = sp->z/ksize;

    // Process 3x3x3 neighborhood with optimized bounds checking
    int zmin = (n.z > 0) ? -1 : 0;
    int zmax = (n.z < lz-1) ? 1 : 0;
    int ymin = (n.y > 0) ? -1 : 0;
    int ymax = (n.y < ly-1) ? 1 : 0;
    int xmin = (n.x > 0) ? -1 : 0;
    int xmax = (n.x < lx-1) ? 1 : 0;

    for(int dz = zmin; dz <= zmax; dz++) {
      int zz = n.z + dz;
      int z_offset = zz * lylx;

      for(int dy = ymin; dy <= ymax; dy++) {
        int yy = n.y + dy;

        for(int dx = xmin; dx <= xmax; dx++) {
          if(dx == 0 && dy == 0 && dz == 0) continue;

          int xx = n.x + dx;
          int ii = z_offset + xx*ly + yy;

          if(labels[ii] <= 0) {
            float dc = scale * (c_over_ksize - img[ii]);
            float dx = x_over_ksize - xx;
            float dy = y_over_ksize - yy;
            float dz = z_over_ksize - zz;
            float d = (dc*dc + (dx*dx + dy*dy + dz*dz)*invwt) / ksize;

            heap_push(&pq, (HeapNode){
              .d = d,
              .k = k,
              .x = (unsigned short)xx,
              .y = (unsigned short)yy,
              .z = (unsigned short)zz
            });
          }
        }
      }
    }
  }

  // Final averaging
  for (unsigned int k = 1; k <= numk; k++) {
    float ksize = (float)superpixels[k].n;
    superpixels[k].c /= ksize;
    superpixels[k].x /= ksize;
    superpixels[k].y /= ksize;
    superpixels[k].z /= ksize;
  }

  heap_free(&pq);
}