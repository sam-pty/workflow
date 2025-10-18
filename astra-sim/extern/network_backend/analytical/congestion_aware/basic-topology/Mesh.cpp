/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Mesh.h"
#include <cassert>
#include <iostream>

using namespace NetworkAnalyticalCongestionAware;

Mesh::Mesh(const int npus_count, const Bandwidth bandwidth, const Latency latency, const bool is_multi_dim) noexcept
    : BasicTopology(npus_count, npus_count, bandwidth, latency) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // set topology type
    this->basic_topology_type = TopologyBuildingBlock::Mesh;

    // connect npus in a mesh
    if (!is_multi_dim) {
        for (auto i = 0; i < npus_count - 1; i++) {
            connect(i, i + 1, bandwidth, latency, true);
        }
    }
}

Route Mesh::route(DeviceId src, DeviceId dest) const noexcept {
    // assert npus are in valid range
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(src != dest);

    // construct empty route
    auto route = Route();
    // std::cout << src << " to " << dest << std::endl;

    if (dest > src)
    {
        for (int i = src; i <= dest; i++)
        {
            route.push_back(devices[i]);
        }
    }
    else
    { // src > dest
        for (int i = src; i >= dest; i--)
        {
            route.push_back(devices[i]);
        }
    }

    // return the constructed route
    return route;
}

std::vector<ConnectionPolicy> Mesh::get_connection_policies() const noexcept {
    std::vector<ConnectionPolicy> policies;

    for (int i = 0; i < npus_count - 1; i++) {
        policies.emplace_back(i, i + 1);
        policies.emplace_back(i + 1, i);
    }

    return policies;
}
