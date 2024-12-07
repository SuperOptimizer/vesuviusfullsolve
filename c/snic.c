#pragma once

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define DIMENSION 256
#define DIMENSION_SHIFT 8  // 2^8 = 256
#define LY_LX (DIMENSION * DIMENSION)
#define IMG_SIZE (DIMENSION * DIMENSION * DIMENSION)

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

static inline int snic_superpixel_count(int d_seed) {
  return (DIMENSION/d_seed) * (DIMENSION/d_seed) * (DIMENSION/d_seed);
}

// Optimize offset calculation for 256x256x256 volume
#define DEFINE_OFFSET(z, y, x) (((z) << (DIMENSION_SHIFT * 2)) | ((x) << DIMENSION_SHIFT) | (y))

#define PROCESS_NEIGHBOR(zoff, yoff, xoff) do { \
  const int zz = n.z + (zoff); \
  const int yy = n.y + (yoff); \
  const int xx = n.x + (xoff); \
  if (zz >= 0 && zz < DIMENSION && \
      yy >= 0 && yy < DIMENSION && \
      xx >= 0 && xx < DIMENSION) { \
    const int ii = DEFINE_OFFSET(zz, yy, xx); \
    if (labels[ii] <= 0) { \
      const float dc = scale * (c_over_ksize - img[ii]); \
      const float dx = x_over_ksize - xx; \
      const float dy = y_over_ksize - yy; \
      const float dz = z_over_ksize - zz; \
      const float d = (dc*dc + (dx*dx + dy*dy + dz*dz)*invwt) / ksize; \
      heap_push(&pq, (HeapNode){ \
        .d = d, \
        .k = k, \
        .x = xx, \
        .y = yy, \
        .z = zz \
      }); \
    } \
  } \
} while(0)

void snic(float *img, int d_seed, float compactness, unsigned int *labels, Superpixel* superpixels) {
  // Pre-calculate constants using exact dimensions
  float spacing = (float)DIMENSION / (float)(DIMENSION/d_seed);

  Heap pq = heap_alloc(IMG_SIZE*16);
  unsigned int numk = 0;

  // Optimize seeding loop for exact dimensions
  for (float iz = spacing/2; iz < DIMENSION; iz += spacing) {
    unsigned char z = (unsigned char)iz;
    for (float iy = spacing/2; iy < DIMENSION; iy += spacing) {
      unsigned char y = (unsigned char)iy;
      for (float ix = spacing/2; ix < DIMENSION; ix += spacing) {
        heap_push(&pq, (HeapNode){
          .d = 0.0f,
          .k = ++numk,
          .x = (unsigned char)ix,
          .y = y,
          .z = z
        });
      }
    }
  }

  float invwt = (compactness*compactness*numk)/(float)(IMG_SIZE);
  const float scale = 100.0f;

  while (pq.len > 0) {
    HeapNode n = heap_pop(&pq);
    const int i = DEFINE_OFFSET(n.z, n.y, n.x);
    if (labels[i] > 0) continue;

    const unsigned int k = n.k;
    labels[i] = k;
    Superpixel* sp = &superpixels[k];
    const float img_val = img[i];
    sp->c += img_val;
    sp->x += n.x;
    sp->y += n.y;
    sp->z += n.z;
    sp->n++;

    const float ksize = (float)sp->n;
    const float c_over_ksize = sp->c/ksize;
    const float x_over_ksize = sp->x/ksize;
    const float y_over_ksize = sp->y/ksize;
    const float z_over_ksize = sp->z/ksize;

    // Process 6 direct neighbors
    PROCESS_NEIGHBOR(-1,  0,  0);
    PROCESS_NEIGHBOR( 1,  0,  0);
    PROCESS_NEIGHBOR( 0, -1,  0);
    PROCESS_NEIGHBOR( 0,  1,  0);
    PROCESS_NEIGHBOR( 0,  0, -1);
    PROCESS_NEIGHBOR( 0,  0,  1);
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