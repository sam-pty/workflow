/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Torus2D.h"
#include <cassert>
#include <cmath>


using namespace NetworkAnalyticalCongestionAware;

Torus2D::Torus2D(const int npus_count,
                 const Bandwidth bandwidth,
                 const Latency latency,
                 const bool bidirectional,
                 const bool is_multi_dim) noexcept
    : bidirectional(bidirectional),
      BasicTopology(npus_count, npus_count, bandwidth, latency, is_multi_dim) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);


    Torus2D::basic_topology_type = TopologyBuildingBlock::Torus2D;

    this->faulty_links = {{4,5}};

    if (!is_multi_dim) {
        // Assume npus_count forms a perfect square
        const int dim = static_cast<int>(std::sqrt(npus_count));
        assert(dim * dim == npus_count && "2D torus requires a square grid");

        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {
                int current = row * dim + col;

                // Connect to right neighbor (wrap around horizontally)
                int right = row * dim + ((col + 1) % dim);
                connect(current, right, bandwidth, latency, bidirectional);

                // Connect to bottom neighbor (wrap around vertically)
                int down = ((row + 1) % dim) * dim + col;
                connect(current, down, bandwidth, latency, bidirectional);
            }
        }
    } else {
        // Fallback to 1D ring
        //for (int i = 0; i < npus_count - 1; ++i) {
        //    connect(i, i + 1, bandwidth, latency, bidirectional);
        //}
        //connect(npus_count - 1, 0, bandwidth, latency, bidirectional);
        std::cerr<<"Torus2D should be defined as a single dimensional topology." << std::endl;
        std::exit(1);
    }

    // this also works
    // std::vector<ConnectionPolicy> policies = get_connection_policies();
    // for (const auto& policy : policies) {
    //     connect(policy.src, policy.dst, bandwidth, latency, /*bidirectional=*/false);
    // }
}

/*

Route Torus2D::route(DeviceId src, DeviceId dest) const noexcept {
    // --- Sanity checks ---
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    Route route;

    const int dim = static_cast<int>(std::sqrt(npus_count));
    assert(dim * dim == npus_count && "2D torus requires perfect square npus_count");

    // --- Compute (x, y) coordinates ---
    auto src_x = src % dim;
    auto src_y = src / dim;
    auto dest_x = dest % dim;
    auto dest_y = dest / dim;

    int dx = dest_x - src_x;
    int dy = dest_y - src_y;

    // Wrap-around distances (for bidirectional torus)
    int wrap_dx = (dx + dim) % dim;
    int wrap_dy = (dy + dim) % dim;


    // --- Choose shortest direction for X ---
    int step_x = 0;
    if (wrap_dx != 0) {
        if (bidirectional && wrap_dx > dim / 2)
            step_x = -1;  // go left
        else
            step_x = +1;  // go right
    }

    // --- Choose shortest direction for Y ---
    int step_y = 0;
    if (wrap_dy != 0) {
        if (bidirectional && wrap_dy > dim / 2)
            step_y = -1;  // go up
        else
            step_y = +1;  // go down
    }

    // --- If the intended X link is faulty, use Y direction instead ---
    int next_x = src + step_x;  // compute target node in X direction
    if (step_x != 0 && is_faulty(src, next_x)) {
        step_y = step_x;
        step_x = 0;  // stop moving in X
        //if (wrap_dy != 0) {
        //    if (bidirectional && wrap_dy > dim / 2)
        //        step_y = -1;  // reroute vertically
        //    else
        //        step_y = +1;
        //}
    }

    // --- If the intended Y link is faulty, use X direction instead ---
    int next_y = src + step_y * dim;  // compute target node in Y direction
    if (step_y != 0 && is_faulty(src, next_y)) {
        step_x = step_y;
        step_y = 0;  // stop moving in Y
        //if (wrap_dx != 0) {
        //    if (bidirectional && wrap_dx > dim / 2)
        //        step_x = -1;  // reroute horizontally
        //    else
        //        step_x = +1;
        //}
    }


    // --- Start routing ---
    int cur_x = src_x;
    int cur_y = src_y;

    // Add starting device
    route.push_back(devices.at(src));

    // --- Dimension-order routing: X first ---
    while (cur_x != dest_x) {
        cur_x = (cur_x + step_x + dim) % dim;
        int idx = cur_y * dim + cur_x;
        route.push_back(devices.at(idx));
    }

    // --- Then route along Y ---
    while (cur_y != dest_y) {
        cur_y = (cur_y + step_y + dim) % dim;
        int idx = cur_y * dim + cur_x;
        route.push_back(devices.at(idx));
    }

    // Done: route includes destination
    return route;
}

*/






Route Torus2D::route(DeviceId src, DeviceId dest) const noexcept {
    Route route;
    const int dim = static_cast<int>(std::sqrt(npus_count));
    int sx = src % dim, sy = src / dim;
    int dx = dest % dim, dy = dest / dim;

    route.push_back(devices.at(src));
    int cur = src;

    while (cur != dest) {
        int cx = cur % dim, cy = cur / dim;

        // Try X direction first
        int step_x = 0, step_y = 0;
        if (cx != dx) {
            int diff_x = (dx - cx + dim) % dim;
            step_x = (diff_x > dim/2) ? -1 : +1;
        } else if (cy != dy) {
            int diff_y = (dy - cy + dim) % dim;
            step_y = (diff_y > dim/2) ? -1 : +1;
        }

        int next;
        if (step_x != 0) {
            int nx = (cx + step_x + dim) % dim;
            next = cy * dim + nx;
            if (is_faulty(cur, next)) {
                // Detour one step in Y
                int ny = (cy + 1) % dim;
                next = ny * dim + cx;
            }
        } else if (step_y != 0) {
            int ny = (cy + step_y + dim) % dim;
            next = ny * dim + cx;
            if (is_faulty(cur, next)) {
                // Detour one step in X
                int nx = (cx + 1) % dim;
                next = cy * dim + nx;
            }
        }

        route.push_back(devices.at(next));
        cur = next;
    }

    return route;
}









std::vector<ConnectionPolicy> Torus2D::get_connection_policies() const noexcept {
    std::vector<ConnectionPolicy> policies;

    const int dim = static_cast<int>(std::sqrt(npus_count));
    assert(dim * dim == npus_count && "2D torus requires npus_count to be a perfect square");

    // Each node connects to its right and down neighbor (with wrap-around)
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            int current = row * dim + col;

            int right = row * dim + ((col + 1) % dim);
            int down  = ((row + 1) % dim) * dim + col;

            // Add unidirectional connections
            policies.emplace_back(current, right);
            policies.emplace_back(current, down);
        }
    }

    // If bidirectional, add reverse edges too
    if (bidirectional) {
        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {
                int current = row * dim + col;

                int left  = row * dim + ((col - 1 + dim) % dim);
                int up    = ((row - 1 + dim) % dim) * dim + col;

                policies.emplace_back(current, left);
                policies.emplace_back(current, up);
            }
        }
    }

    return policies;
}

bool Torus2D::is_faulty(int src, int dst) const {
        for (auto &link : faulty_links) {
            if ((link.first == src && link.second == dst) ||
                (link.first == dst && link.second == src)) {
                return true;
            }
        }
        return false;
    };

    

