/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Mesh2D.h"
#include <cassert>
#include <cmath>


using namespace NetworkAnalyticalCongestionAware;

Mesh2D::Mesh2D(const int npus_count,
                 const Bandwidth bandwidth,
                 const Latency latency,
                 const bool bidirectional,
                 const bool is_multi_dim) noexcept
    : bidirectional(bidirectional),
      BasicTopology(npus_count, npus_count, bandwidth, latency, is_multi_dim) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    Mesh2D::basic_topology_type = TopologyBuildingBlock::Mesh2D;

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

Route Mesh2D::route(DeviceId src, DeviceId dest) const noexcept {
    // --- Sanity checks ---
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    Route route;

    const int dim = static_cast<int>(std::sqrt(npus_count));
    assert(dim * dim == npus_count && "2D mesh requires perfect square npus_count");

    // --- Compute (x, y) coordinates ---
    int src_x = src % dim;
    int src_y = src / dim;
    int dest_x = dest % dim;
    int dest_y = dest / dim;

    // --- Compute deltas ---
    int dx = dest_x - src_x;
    int dy = dest_y - src_y;

    // --- Determine step directions ---
    int step_x = (dx == 0) ? 0 : (dx > 0 ? +1 : -1);
    int step_y = (dy == 0) ? 0 : (dy > 0 ? +1 : -1);

    // --- Start from source ---
    int cur_x = src_x;
    int cur_y = src_y;

    route.push_back(devices.at(src));

    // --- Route along X first ---
    while (cur_x != dest_x) {
        cur_x += step_x;
        int idx = cur_y * dim + cur_x;
        route.push_back(devices.at(idx));
    }

    // --- Then along Y ---
    while (cur_y != dest_y) {
        cur_y += step_y;
        int idx = cur_y * dim + cur_x;
        route.push_back(devices.at(idx));
    }

    // --- Done ---
    return route;
}

std::vector<ConnectionPolicy> Mesh2D::get_connection_policies() const noexcept {
    std::vector<ConnectionPolicy> policies;

    const int dim = static_cast<int>(std::sqrt(npus_count));
    assert(dim * dim == npus_count && "2D mesh requires npus_count to be a perfect square");

    // Each node connects to its right and down neighbor (no wrap-around)
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            int current = row * dim + col;

            // Connect to right neighbor (if not on the right edge)
            if (col + 1 < dim) {
                int right = row * dim + (col + 1);
                policies.emplace_back(current, right);
            }

            // Connect to down neighbor (if not on the bottom edge)
            if (row + 1 < dim) {
                int down = (row + 1) * dim + col;
                policies.emplace_back(current, down);
            }
        }
    }

    // If bidirectional, add reverse edges too
    if (bidirectional) {
        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {
                int current = row * dim + col;

                // Connect to left neighbor (if not on the left edge)
                if (col - 1 >= 0) {
                    int left = row * dim + (col - 1);
                    policies.emplace_back(current, left);
                }

                // Connect to up neighbor (if not on the top edge)
                if (row - 1 >= 0) {
                    int up = (row - 1) * dim + col;
                    policies.emplace_back(current, up);
                }
            }
        }
    }

    return policies;
}

