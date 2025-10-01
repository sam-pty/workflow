/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Ring.h"
#include <cassert>

using namespace NetworkAnalyticalCongestionAware;

Ring::Ring(const int npus_count, const Bandwidth bandwidth, const Latency latency, const bool bidirectional) noexcept
    : bidirectional(bidirectional),
      BasicTopology(npus_count, npus_count, bandwidth, latency) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // set topology type
    Ring::basic_topology_type = TopologyBuildingBlock::Ring;

    // connect npus in a ring
    for (auto i = 0; i < npus_count - 1; i++) {
        connect(i, i + 1, bandwidth, latency, bidirectional);
    }
    connect(npus_count - 1, 0, bandwidth, latency, bidirectional);

    // this also works
    // std::vector<ConnectionPolicy> policies = get_connection_policies();
    // for (const auto& policy : policies) {
    //     connect(policy.src, policy.dst, bandwidth, latency, /*bidirectional=*/false);
    // }
}

Route Ring::route(DeviceId src, DeviceId dest) const noexcept {
    // assert npus are in valid range
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    // construct empty route
    auto route = Route();

    auto step = 1;  // default direction: clockwise
    if (bidirectional) {
        // check whether going anticlockwise is shorter
        auto clockwise_dist = dest - src;
        if (clockwise_dist < 0) {
            clockwise_dist += npus_count;
        }
        const auto anticlockwise_dist = npus_count - clockwise_dist;

        if (anticlockwise_dist < clockwise_dist) {
            // traverse the ring anticlockwise
            step = -1;
        }
    }

    // construct the route
    auto current = src;
    while (current != dest) {
        // traverse the ring until reaches dest
        route.push_back(devices.at(current));
        current = (current + step);

        // wrap around
        if (current < 0) {
            current += npus_count;
        } else if (current >= npus_count) {
            current -= npus_count;
        }
    }

    // arrives at dest
    route.push_back(devices.at(dest));

    // return the constructed route
    return route;
}

std::vector<ConnectionPolicy> Ring::get_connection_policies() const noexcept {
    std::vector<ConnectionPolicy> policies;

    for (int i = 0; i < npus_count; i++) {
        policies.emplace_back(ConnectionPolicy{i, (i + 1) % npus_count});
    }

    if (bidirectional) {
        for (int i = 0; i < npus_count; i++) {
            policies.emplace_back(ConnectionPolicy{(i + 1) % npus_count, i});
        }
    }

    return policies;
}
