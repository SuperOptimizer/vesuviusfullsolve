
// chord.c
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "chord.h"

typedef struct Vec3f {
    f32 x, y, z;
} Vec3f;

static f32 vec3f_dot(Vec3f a, Vec3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static f32 vec3f_length(Vec3f v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static Vec3f vec3f_normalize(Vec3f v) {
    f32 len = vec3f_length(v);
    if (len > 0) {
        return (Vec3f){v.x / len, v.y / len, v.z / len};
    }
    return (Vec3f){0, 0, 1};  // Default to up direction
}

static Vec3f vec3f_sub(Vec3f a, Vec3f b) {
    return (Vec3f){a.x - b.x, a.y - b.y, a.z - b.z};
}

static f32 vec3f_dist(Vec3f a, Vec3f b) {
    Vec3f diff = vec3f_sub(a, b);
    return vec3f_length(diff);
}

// Find nearest points to a position within search radius
static void find_nearby_points(
    const Superpixel* superpixels,
    u32 num_superpixels,
    Vec3f pos,
    f32 radius,
    const bool* used,
    u32* nearby_indices,
    u32* num_nearby
) {
    *num_nearby = 0;
    for (u32 i = 1; i <= num_superpixels; i++) {
        if (used[i]) continue;

        Vec3f sp_pos = {superpixels[i].x, superpixels[i].y, superpixels[i].z};
        f32 dist = vec3f_dist(pos, sp_pos);

        if (dist <= radius) {
            nearby_indices[*num_nearby] = i;
            (*num_nearby)++;
        }
    }
}

static Vec3f estimate_gradient(
    const Superpixel* superpixels,
    Vec3f pos,
    const u32* nearby,
    u32 num_nearby,
    f32 z_bias
) {
    Vec3f gradient = {0, 0, 0};

    for (u32 i = 0; i < num_nearby; i++) {
        u32 idx = nearby[i];
        Vec3f sp_pos = {superpixels[idx].x, superpixels[idx].y, superpixels[idx].z};
        Vec3f diff = vec3f_sub(sp_pos, pos);
        f32 dist = vec3f_length(diff);
        if (dist < 0.001f) continue;

        f32 weight = superpixels[idx].c / (dist + 0.001f);
        gradient.x += weight * diff.x / dist;
        gradient.y += weight * diff.y / dist;
        gradient.z += weight * diff.z / dist;
    }

    // Add z-bias
    gradient.z += z_bias * 10.0f;

    return vec3f_normalize(gradient);
}

ChordResult* grow_chords(
    const Superpixel* superpixels,
    u32 num_superpixels,
    u32 num_chords,
    u32 min_length,
    u32 max_length,
    f32 z_bias,
    f32 min_direction_score,
    f32 search_radius
) {
    ChordResult* result = malloc(sizeof(ChordResult));
    result->chords = malloc(num_chords * sizeof(Chord));
    result->num_chords = 0;

    // Allocate memory for tracking used points
    bool* used = calloc(num_superpixels + 1, sizeof(bool));

    // Temporary storage for nearby points
    u32* nearby = malloc(num_superpixels * sizeof(u32));

    // Find starting points near bottom of volume
    u32* starting_points = malloc(num_chords * sizeof(u32));
    u32 num_starting = 0;
    f32 z_min = INFINITY;

    // Find minimum z coordinate
    for (u32 i = 1; i <= num_superpixels; i++) {
        if (superpixels[i].z < z_min) {
            z_min = superpixels[i].z;
        }
    }

    // Find starting points near bottom
    f32 z_tolerance = 5.0f;
    while (num_starting < num_chords && z_tolerance < 100.0f) {
        for (u32 i = 1; i <= num_superpixels && num_starting < num_chords; i++) {
            if (!used[i] && fabsf(superpixels[i].z - z_min) <= z_tolerance) {
                starting_points[num_starting++] = i;
                used[i] = true;
            }
        }
        z_tolerance *= 1.5f;
    }

    // Grow chords from starting points
    for (u32 i = 0; i < num_starting; i++) {
        Chord* chord = &result->chords[result->num_chords];
        chord->point_labels = malloc(max_length * sizeof(u32));
        chord->length = 0;

        u32 current_label = starting_points[i];
        Vec3f current_pos = {
            superpixels[current_label].x,
            superpixels[current_label].y,
            superpixels[current_label].z
        };

        // Add starting point
        chord->point_labels[chord->length++] = current_label;

        // Grow chord
        while (chord->length < max_length) {
            u32 num_nearby;
            find_nearby_points(superpixels, num_superpixels, current_pos,
                             search_radius, used, nearby, &num_nearby);

            if (num_nearby == 0) break;

            // Estimate gradient direction
            Vec3f gradient = estimate_gradient(superpixels, current_pos,
                                            nearby, num_nearby, z_bias);

            // Find best next point
            f32 best_score = -1;
            u32 best_idx = 0;

            for (u32 j = 0; j < num_nearby; j++) {
                u32 idx = nearby[j];
                Vec3f sp_pos = {superpixels[idx].x, superpixels[idx].y, superpixels[idx].z};
                Vec3f dir = vec3f_normalize(vec3f_sub(sp_pos, current_pos));

                f32 direction_score = vec3f_dot(dir, gradient);
                if (direction_score > best_score) {
                    best_score = direction_score;
                    best_idx = idx;
                }
            }

            if (best_score < min_direction_score) break;

            // Add best point to chord
            current_label = best_idx;
            current_pos = (Vec3f){
                superpixels[current_label].x,
                superpixels[current_label].y,
                superpixels[current_label].z
            };
            chord->point_labels[chord->length++] = current_label;
            used[current_label] = true;
        }

        // Only keep chord if it meets minimum length
        if (chord->length >= min_length) {
            result->num_chords++;
        } else {
            free(chord->point_labels);
        }
    }

    // Cleanup
    free(used);
    free(nearby);
    free(starting_points);

    return result;
}

void free_chord_result(ChordResult* result) {
    if (!result) return;
    for (u32 i = 0; i < result->num_chords; i++) {
        free(result->chords[i].point_labels);
    }
    free(result->chords);
    free(result);
}