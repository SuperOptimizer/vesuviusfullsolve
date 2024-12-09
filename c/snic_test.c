#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include "snic.h"

#define LZ 256
#define LY 256
#define LX 256
#define D_SEED 8
#define COMPACTNESS ((f32)(D_SEED*D_SEED*D_SEED))  // Typical value for SNIC compactness

void generate_test_volume(uint8_t* volume) {
    for(int z = 0; z < LZ; z++) {
        for(int y = 0; y < LY; y++) {
            for(int x = 0; x < LX; x++) {
                if((x ^ y ^ z) & 32) {
                    volume[z*LY*LX + y*LX + x] =
                        (uint8_t)((x + y + z) / 3);
                } else {
                    volume[z*LY*LX + y*LX + x] =
                        (uint8_t)(255 - (x + y + z) / 3);
                }
            }
        }
    }
}

void verify_labels(uint32_t* labels) {
    int unlabeled = 0;
    uint32_t max_label = 0;

    for(int i = 0; i < LZ*LY*LX; i++) {
        if(labels[i] == 0) unlabeled++;
        if(labels[i] > max_label) max_label = labels[i];
    }
    int total_voxels = LZ*LY*LX;
    printf("Verification:\n");
    printf("- Unlabeled voxels: %d\n", unlabeled);
    printf("- Max label: %u\n", max_label);
    printf("- Expected labels: %d\n",
           (LZ/D_SEED)*(LY/D_SEED)*(LX/D_SEED));
    int expected_clusters = (LZ/D_SEED)*(LY/D_SEED)*(LX/D_SEED);
    printf("Verification:\n");
    printf("- Volume size: %d x %d x %d\n", LZ, LY, LX);
    printf("- Grid size: %d x %d x %d\n", LZ/D_SEED, LY/D_SEED, LX/D_SEED);
    printf("- Unlabeled voxels: %d (%.2f%%)\n",
           unlabeled, 100.0f * unlabeled / total_voxels);
    printf("- Max label: %u\n", max_label);
    printf("- Expected clusters: %d\n", expected_clusters);
}

int main() {
    // Allocate memory with alignment for better performance
    uint8_t* volume = aligned_alloc(32, LZ*LY*LX);
    uint32_t* labels = aligned_alloc(32, LZ*LY*LX*sizeof(uint32_t));
    Superpixel* superpixels = calloc(
        (LZ/D_SEED)*(LY/D_SEED)*(LX/D_SEED) + 1,
        sizeof(Superpixel));

    if(!volume || !labels || !superpixels) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize labels to zero
    memset(labels, 0, LZ*LY*LX*sizeof(uint32_t));

    // Generate test volume
    generate_test_volume(volume);

    // Time the SNIC execution
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int neigh_overflow = snic(volume, LZ, LY, LX, D_SEED, COMPACTNESS, labels, superpixels);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time: %.3f seconds\n", time);
    printf("Neighbor overflow: %d\n", neigh_overflow);

    verify_labels(labels);

    free(volume);
    free(labels);
    free(superpixels);
    return 0;
}