#pragma once

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define ISO_THRESHOLD 1
#define D_SEED 2
#define DIMENSION 256
#define DIMENSION_SHIFT 8
#define LY_LX (DIMENSION * DIMENSION)
#define IMG_SIZE (DIMENSION * DIMENSION * DIMENSION)

typedef struct HeapNode {
    uint32_t k;
    uint8_t x, y, z;
        uint8_t d;

} HeapNode __attribute__((aligned(8)));

#define heap_node_val(n) (255 - n.d)

typedef struct Superpixel {
    uint8_t x, y, z;
    uint8_t c;
    uint32_t n;
} Superpixel __attribute__((aligned(8)));

typedef struct Heap {
    int32_t len;
    int32_t size;
    HeapNode nodes[];  // Flexible array member
} Heap __attribute__((aligned(8)));

static inline Heap* heap_alloc(int32_t size) {
    Heap* heap = (Heap*)malloc(sizeof(Heap) + size * sizeof(HeapNode));
    heap->len = 0;
    heap->size = size;
    return heap;
}

static inline void heap_free(Heap *heap) {
    free(heap);
}

// Update other heap functions to work with Heap* instead of Heap
static inline void heap_push(Heap *heap, HeapNode node) {
    int32_t i = ++heap->len;

    for (; i > 1 && heap_node_val(heap->nodes[i >> 1]) < heap_node_val(node); i >>= 1) {
        heap->nodes[i] = heap->nodes[i >> 1];
    }
    heap->nodes[i] = node;
}

static inline HeapNode heap_pop(Heap *heap) {
    HeapNode result = heap->nodes[1];
    HeapNode last = heap->nodes[heap->len--];
    int32_t i = 1, child;

    while ((child = i << 1) <= heap->len) {
        if (child < heap->len && heap_node_val(heap->nodes[child]) < heap_node_val(heap->nodes[child + 1])) {
            child++;
        }
        if (heap_node_val(last) >= heap_node_val(heap->nodes[child])) break;
        heap->nodes[i] = heap->nodes[child];
        i = child;
    }
    heap->nodes[i] = last;
    return result;
}

#define OFFSET(z, y, x) (((uint32_t)(z) << (DIMENSION_SHIFT * 2)) | ((uint32_t)(x) << DIMENSION_SHIFT) | (y))

static inline void process_neighbor(Heap* pq, const uint8_t* img, uint32_t* labels,
                                  uint32_t k, float scale, float invwt, float ksize,
                                  float c_over_ksize, float x_over_ksize, float y_over_ksize, float z_over_ksize,
                                  int8_t zoff, int8_t yoff, int8_t xoff, uint8_t cz, uint8_t cy, uint8_t cx) {
    const int32_t zz = cz + zoff;
    const int32_t yy = cy + yoff;
    const int32_t xx = cx + xoff;

    if (zz >= 0 && zz < DIMENSION && yy >= 0 && yy < DIMENSION && xx >= 0 && xx < DIMENSION) {
        const uint32_t ii = OFFSET(zz, yy, xx);
        if (labels[ii] == 0) {
            const float dc = scale * (c_over_ksize - img[ii]);
            const float dx = x_over_ksize - xx;
            const float dy = y_over_ksize - yy;
            const float dz = z_over_ksize - zz;
            const float d = (dc * dc + (dx * dx + dy * dy + dz * dz) * invwt) / ksize;

            // Scale and clamp distance to 0-255 range
            const uint8_t d_scaled = (uint8_t)fminf(255.0f, d * 64.0f);
            heap_push(pq, (HeapNode){d_scaled, k, xx, yy, zz});
        }
    }
}

uint32_t snic(uint8_t* img, uint32_t* labels, Superpixel* superpixels) {
    const float spacing = (float)DIMENSION / (DIMENSION/D_SEED);
    const float invwt = ((DIMENSION/D_SEED) * (DIMENSION/D_SEED) * (DIMENSION/D_SEED))/(float)(IMG_SIZE);
    const float scale = 1.0f;
    Heap* pq = heap_alloc(IMG_SIZE >> 2);
    uint32_t k = 0;

    for (uint32_t i = 0; i < IMG_SIZE; i++) {
        labels[i] = 0;
    }
    superpixels[0] = (Superpixel){0};

    for (float iz = spacing/2; iz < DIMENSION; iz += spacing) {
        const uint8_t z = (uint8_t)iz;
        for (float iy = spacing/2; iy < DIMENSION; iy += spacing) {
            const uint8_t y = (uint8_t)iy;
            for (float ix = spacing/2; ix < DIMENSION; ix += spacing) {
                const uint8_t x = (uint8_t)ix;
                const uint32_t i = OFFSET(z, y, x);

                if (img[i] >= ISO_THRESHOLD) {
                    heap_push(pq, (HeapNode){0, ++k, x, y, z});
                    superpixels[k] = (Superpixel){x, y, z, img[i], 1};
                }
            }
        }
    }

    while (pq->len > 0) {
        const HeapNode n = heap_pop(pq);
        const uint32_t i = OFFSET(n.z, n.y, n.x);

        if (labels[i] > 0) continue;

        const uint32_t k = n.k;
        labels[i] = k;
        Superpixel* sp = &superpixels[k];
        const uint8_t img_val = img[i];

        if ((sp->c * sp->n + img_val) / (sp->n + 1) < ISO_THRESHOLD) continue;

        sp->c = (sp->c * sp->n + img_val) / (sp->n + 1);
        sp->x = n.x;
        sp->y = n.y;
        sp->z = n.z;
        sp->n++;

        const float ksize = (float)sp->n;
        const float c_over_ksize = sp->c;
        const float x_over_ksize = sp->x;
        const float y_over_ksize = sp->y;
        const float z_over_ksize = sp->z;

        static const int8_t offsets[6][3] = {{-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}};
        for (int j = 0; j < 6; j++) {
            process_neighbor(pq, img, labels, k, scale, invwt, ksize,
                           c_over_ksize, x_over_ksize, y_over_ksize, z_over_ksize,
                           offsets[j][0], offsets[j][1], offsets[j][2], n.z, n.y, n.x);
        }
    }

    heap_free(pq);
    return k;
}