#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include "snic.h"
#include "chord.h"

#define LZ 256
#define LY 256
#define LX 256
#define D_SEED 8
#define COMPACTNESS ((f32)(D_SEED*D_SEED*D_SEED))

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
    int expected_clusters = (LZ/D_SEED)*(LY/D_SEED)*(LX/D_SEED);

    printf("SNIC Verification:\n");
    printf("- Volume size: %d x %d x %d\n", LZ, LY, LX);
    printf("- Grid size: %d x %d x %d\n", LZ/D_SEED, LY/D_SEED, LX/D_SEED);
    printf("- Unlabeled voxels: %d (%.2f%%)\n",
           unlabeled, 100.0f * unlabeled / total_voxels);
    printf("- Max label: %u\n", max_label);
    printf("- Expected clusters: %d\n", expected_clusters);
}

void verify_chords(ChordResult* chord_result) {
    if (!chord_result) {
        printf("Chord verification failed: null result\n");
        return;
    }

    printf("\nChord Verification:\n");
    printf("- Total chords: %u\n", chord_result->num_chords);

    if (chord_result->num_chords > 0) {
        u32 min_length = UINT32_MAX;
        u32 max_length = 0;
        f32 avg_length = 0;
        u32 total_points = 0;

        for (u32 i = 0; i < chord_result->num_chords; i++) {
            u32 len = chord_result->chords[i].length;
            if (len < min_length) min_length = len;
            if (len > max_length) max_length = len;
            avg_length += len;
            total_points += len;
        }

        avg_length /= chord_result->num_chords;

        printf("- Length statistics:\n");
        printf("  - Minimum: %u\n", min_length);
        printf("  - Maximum: %u\n", max_length);
        printf("  - Average: %.1f\n", avg_length);
        printf("  - Total points used: %u\n", total_points);
    }
}

int main() {
    // Allocate memory
    uint8_t* volume = aligned_alloc(32, LZ*LY*LX);
    uint32_t* labels = aligned_alloc(32, LZ*LY*LX*sizeof(uint32_t));
    u32 max_superpixels = (LZ/D_SEED)*(LY/D_SEED)*(LX/D_SEED) + 1;
    Superpixel* superpixels = calloc(max_superpixels, sizeof(Superpixel));

    if(!volume || !labels || !superpixels) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize labels to zero
    memset(labels, 0, LZ*LY*LX*sizeof(uint32_t));

    // Generate test volume
    generate_test_volume(volume);

    // Time SNIC execution
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int neigh_overflow = snic(volume, LZ, LY, LX, D_SEED, COMPACTNESS, labels, superpixels);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double snic_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("SNIC Time: %.3f seconds\n", snic_time);
    printf("Neighbor overflow: %d\n", neigh_overflow);
    verify_labels(labels);

    // Grow chords from superpixels
    printf("\nStarting chord growth...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);

    ChordResult* chord_result = grow_chords(
        superpixels,
        max_superpixels - 1,  // Exclude sentinel superpixel
        4096,                 // Number of chords to attempt
        16,                   // Minimum chord length
        64,                   // Maximum chord length
        0.5f,                // Z-direction bias
        0.3f,                // Minimum direction score
        8.0f                 // Search radius
    );

    clock_gettime(CLOCK_MONOTONIC, &end);
    double chord_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Chord Growth Time: %.3f seconds\n", chord_time);
    verify_chords(chord_result);

    // Print total processing time
    printf("\nTotal Processing Time: %.3f seconds\n", snic_time + chord_time);

    // Cleanup
    free(volume);
    free(labels);
    free(superpixels);
    if (chord_result) {
        free_chord_result(chord_result);
    }

    return 0;
}