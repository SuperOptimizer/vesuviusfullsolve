#pragma once

#include "common.h"
#define SUPERPIXEL_MAX_NEIGHS 27


typedef struct Superpixel {
  f32 x, y, z, c;
  u32 n;
  u32 neighs[SUPERPIXEL_MAX_NEIGHS];
} Superpixel;

int snic(const u8* img, u16 lz, u16 ly, u16 lx, u32 d_seed, f32 compactness, u32* labels, Superpixel* superpixels);