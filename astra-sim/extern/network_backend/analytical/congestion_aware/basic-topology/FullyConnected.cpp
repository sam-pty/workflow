/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/FullyConnected.h"
#include <cassert>

using namespace NetworkAnalyticalCongestionAware;

FullyConnected::FullyConnected(const int npus_count,
                               const Bandwidth bandwidth,
                               const Latency latency,
                               const bool is_multi_dim) noexcept
    : BasicTopology(npus_count, npus_count, bandwidth, latency) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // set topology type
    basic_topology_type = TopologyBuildingBlock::FullyConnected;

    // fully-connect every src-dest pairs
    if (!is_multi_dim) {
        for (auto src = 0; src < npus_count; src++) {
            for (auto dest = 0; dest < npus_count; dest++) {
                if (src != dest) {
                    connect(src, dest, bandwidth, latency, false);
                }
            }
        }
    }
}

Route FullyConnected::route(const DeviceId src, const DeviceId dest) const noexcept {
    // assert npus are in valid range
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    // construct route
    // directly connected
    auto route = Route();
    route.push_back(devices[src]);
    route.push_back(devices[dest]);

    return route;
}

std::vector<ConnectionPolicy> FullyConnected::get_connection_policies() const noexcept {
    std::vector<ConnectionPolicy> policies;

    for (auto src = 0; src < npus_count; src++) {
        for (auto dest = 0; dest < npus_count; dest++) {
            if (src != dest) {
                policies.emplace_back(src, dest);
            }
        }
    }

    return policies;
}
