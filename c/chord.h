// chord.h
#pragma once

#include "common.h"
#include "snic.h"
#include <stdbool.h>

typedef struct ChordPoint {
    u32 label;  // Original superpixel label
    bool used;  // Whether this point has been used in a chord
} ChordPoint;

typedef struct Chord {
    u32* point_labels;  // Array of superpixel labels in this chord
    u32 length;
} Chord;

typedef struct ChordResult {
    Chord* chords;
    u32 num_chords;
} ChordResult;

ChordResult* grow_chords(
    const Superpixel* superpixels,
    u32 num_superpixels,
    u32 num_chords,
    u32 min_length,
    u32 max_length,
    f32 z_bias,
    f32 min_direction_score,
    f32 search_radius
);

void free_chord_result(ChordResult* result);