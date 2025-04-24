#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <cstdint>
#include <array>
#include <vector>
#include <queue>
#include <google/dense_hash_map>


int ballquery_batch_p(at::Tensor xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor batch_offsets_tensor, at::Tensor idx_tensor, at::Tensor start_len_tensor, int n, int meanActive, float radius);

void bfs_cluster(at::Tensor semantic_label_tensor, at::Tensor ball_query_idxs_tensor, at::Tensor start_len_tensor,
at::Tensor cluster_idxs_tensor, at::Tensor cluster_offsets_tensor, const int N, int threshold);