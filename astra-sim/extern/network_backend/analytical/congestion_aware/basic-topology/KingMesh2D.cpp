/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/KingMesh2D.h"
#include <cassert>
#include <cmath>


using namespace NetworkAnalyticalCongestionAware;

KingMesh2D::KingMesh2D(const int npus_count,
                 const Bandwidth bandwidth,
                 const Latency latency,
                 const bool bidirectional,
                 const bool is_multi_dim) noexcept
    : bidirectional(bidirectional),
      BasicTopology(npus_count, npus_count, bandwidth, latency, is_multi_dim) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    KingMesh2D::basic_topology_type = TopologyBuildingBlock::KingMesh2D;

    if (!is_multi_dim) {
        // Assume npus_count forms a perfect square
        const int dim = static_cast<int>(std::sqrt(npus_count));
        assert(dim * dim == npus_count && "2D Mesh requires a square grid");

        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {
                int current = row * dim + col;

                // --- Connect right (no wrap-around) ---
                if (col + 1 < dim) {
                    int right = row * dim + (col + 1);
                    connect(current, right, bandwidth, latency, bidirectional);
                } 

                // --- Connect down (no wrap-around) ---
                if (row + 1 < dim) {
                    int down = (row + 1) * dim + col;
                    connect(current, down, bandwidth, latency, bidirectional);
                    if (col + 1 < dim){
                        int diag_right = (row + 1) * dim + (col + 1);
                        connect(current, diag_right, bandwidth, latency, bidirectional);
                    }
                    if (col > 0){
                        int diag_left = (row + 1) * dim + (col - 1);
                        connect(current, diag_left, bandwidth, latency, bidirectional);
                    }
                }
            }
        }
    } else {
        // Fallback to 1D Mesh
        for (int i = 0; i < npus_count - 1; ++i) {
            connect(i, i + 1, bandwidth, latency, bidirectional);
        }
        //connect(npus_count - 1, 0, bandwidth, latency, bidirectional);
    }

    // this also works
    // std::vector<ConnectionPolicy> policies = get_connection_policies();
    // for (const auto& policy : policies) {
    //     connect(policy.src, policy.dst, bandwidth, latency, /*bidirectional=*/false);
    // }
}

Route KingMesh2D::route(DeviceId src, DeviceId dest) const noexcept {
    // --- Sanity checks ---
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    Route route;

    const int dim = static_cast<int>(std::sqrt(npus_count));
    assert(dim * dim == npus_count && "KingMesh2D requires perfect square npus_count");

    // --- Compute (x, y) coordinates ---
    int src_x = src % dim;
    int src_y = src / dim;
    int dest_x = dest % dim;
    int dest_y = dest / dim;

    int cur_x = src_x;
    int cur_y = src_y;

    route.push_back(devices.at(src));

    // --- Diagonal + dimension-order routing ---
    while (cur_x != dest_x || cur_y != dest_y) {
        int step_x = 0, step_y = 0;

        if (cur_x < dest_x) step_x = +1;
        else if (cur_x > dest_x) step_x = -1;

        if (cur_y < dest_y) step_y = +1;
        else if (cur_y > dest_y) step_y = -1;

        // Move diagonally if both deltas are nonzero
        cur_x += step_x;
        cur_y += step_y;

        // Clamp to mesh boundaries (just in case)
        if (cur_x < 0) cur_x = 0;
        else if (cur_x >= dim) cur_x = dim - 1;

if (cur_y < 0) cur_y = 0;
else if (cur_y >= dim) cur_y = dim - 1;

        int idx = cur_y * dim + cur_x;
        route.push_back(devices.at(idx));
    }

    return route;
}


std::vector<ConnectionPolicy> KingMesh2D::get_connection_policies() const noexcept {
    std::vector<ConnectionPolicy> policies;

    const int dim = static_cast<int>(std::sqrt(npus_count));
    assert(dim * dim == npus_count && "KingMesh2D requires npus_count to be a perfect square");

    // --- Each node connects to its 8-neighborhood (no wrap-around) ---
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            int current = row * dim + col;

            // Explore all 8 directions
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    // Skip self
                    if (dx == 0 && dy == 0) continue;

                    int new_row = row + dy;
                    int new_col = col + dx;

                    // Boundary check (no wrap-around)
                    if (new_row >= 0 && new_row < dim && new_col >= 0 && new_col < dim) {
                        int neighbor = new_row * dim + new_col;
                        policies.emplace_back(current, neighbor);
                    }
                }
            }
        }
    }

    // --- If bidirectional, add reverse edges explicitly ---
    if (bidirectional) {
        std::vector<ConnectionPolicy> reverse_policies;
        reverse_policies.reserve(policies.size());

        for (const auto& p : policies) {
            reverse_policies.emplace_back(p.dst, p.src);
        }

        // Append reverse edges
        policies.insert(policies.end(), reverse_policies.begin(), reverse_policies.end());
    }

    return policies;
}


