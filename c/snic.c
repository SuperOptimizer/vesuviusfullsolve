#pragma once

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define D_SEED 2
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

typedef struct Superpixel {
    unsigned char x, y, z;  // Coordinates as uint8
    unsigned char c;  // Changed from float to uint8
    unsigned int n;
} Superpixel;

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

static inline int snic_superpixel_count(int d_seed) {
    return (DIMENSION/D_SEED) * (DIMENSION/D_SEED) * (DIMENSION/D_SEED);
}

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

unsigned int snic(unsigned char *img, float compactness, unsigned int *labels, Superpixel* superpixels, unsigned char iso_threshold) {
    float spacing = (float)DIMENSION / (float)(DIMENSION/D_SEED);
    Heap pq = heap_alloc(IMG_SIZE*16);
    const unsigned int numk = snic_superpixel_count(D_SEED);

    for (int i = 0; i < IMG_SIZE; i++) {
        labels[i] = 0;
    }

    superpixels[0] = (Superpixel){0};

    unsigned int k = 0;
    for (float iz = spacing/2; iz < DIMENSION; iz += spacing) {
        unsigned char z = (unsigned char)iz;
        for (float iy = spacing/2; iy < DIMENSION; iy += spacing) {
            unsigned char y = (unsigned char)iy;
            for (float ix = spacing/2; ix < DIMENSION; ix += spacing) {
                heap_push(&pq, (HeapNode){
                    .d = 0.0f,
                    .k = ++k,
                    .x = (unsigned char)ix,
                    .y = y,
                    .z = z
                });
            }
        }
    }

    const  float invwt = ((DIMENSION/D_SEED) * (DIMENSION/D_SEED) * (DIMENSION/D_SEED))/(float)(IMG_SIZE);
    const float scale = 1.0f;

    while (pq.len > 0) {
        HeapNode n = heap_pop(&pq);
        const int i = DEFINE_OFFSET(n.z, n.y, n.x);
        if (labels[i] > 0) continue;

        const unsigned int k = n.k;
        labels[i] = k;
        Superpixel* sp = &superpixels[k];
        const unsigned char img_val = img[i];
        sp->c = (sp->c * sp->n + img_val) / (sp->n + 1);
        sp->x = n.x;
        sp->y = n.y;
        sp->z = n.z;
        sp->n++;

        const float ksize = (float)sp->n;
        const float c_over_ksize = (float)sp->c;
        const float x_over_ksize = sp->x;
        const float y_over_ksize = sp->y;
        const float z_over_ksize = sp->z;

        PROCESS_NEIGHBOR(-1,  0,  0);
        PROCESS_NEIGHBOR( 1,  0,  0);
        PROCESS_NEIGHBOR( 0, -1,  0);
        PROCESS_NEIGHBOR( 0,  1,  0);
        PROCESS_NEIGHBOR( 0,  0, -1);
        PROCESS_NEIGHBOR( 0,  0,  1);
    }

    heap_free(&pq);

    unsigned int new_label = 1;
    unsigned int* label_map = (unsigned int*)calloc(numk + 1, sizeof(unsigned int));

    for (unsigned int k = 1; k <= numk; k++) {
        if (superpixels[k].c >= iso_threshold) {
            label_map[k] = new_label++;
            superpixels[label_map[k]] = superpixels[k];
        }
    }

    for (unsigned int k = new_label; k <= numk; k++) {
        superpixels[k] = (Superpixel){0};
    }

    for (int i = 0; i < IMG_SIZE; i++) {
        unsigned int old_label = labels[i];
        labels[i] = (old_label > 0) ? label_map[old_label] : 0;
    }

    free(label_map);
    return new_label - 1;
}