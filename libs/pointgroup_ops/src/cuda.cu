#include <ATen/ATen.h>
#include "datatype/datatype.h"

#include "voxelize/voxelize.cu"

template void voxelize_fp_cuda<float>(Int nOutputRows, Int maxActive, Int nPlanes, float *feats, float *output_feats, Int *rules, bool average);

template void voxelize_bp_cuda<float>(Int nOutputRows, Int maxActive, Int nPlanes, float *d_output_feats, float *d_feats, Int *rules, bool average);
