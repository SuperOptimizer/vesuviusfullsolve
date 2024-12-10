#pragma once

#include "common.h"

#define MAX_NEIGHBORS 4

typedef struct Superpixel {
  f32 x, y, z, c;
  u32 n;
} Superpixel;

int snic(const u8* img, u16 lz, u16 ly, u16 lx, u32 d_seed, f32 compactness, u32* labels, Superpixel* superpixels);