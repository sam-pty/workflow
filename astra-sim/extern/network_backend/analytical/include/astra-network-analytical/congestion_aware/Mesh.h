/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/BasicTopology.h"

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * Implements a 1-D mesh topology.
 *
 * Mesh(4) example:
 * 0 - 1 - 2 - 3
 *
 * mesh has to be bi-directional, each chunk can flow through:
 * 0 -> 1 -> 2 -> 3
 * 0 <- 1 <- 2 <- 3
 */
class Mesh final : public BasicTopology {
  public:
    /**
     * Constructor.
     *
     * @param npus_count number of npus in a mesh
     * @param bandwidth bandwidth of link
     * @param latency latency of link
     * @param bidirectional true if mesh is bidirectional, false otherwise
     */
    Mesh(int npus_count, Bandwidth bandwidth, Latency latency, bool is_multi_dim = false) noexcept;

    /**
     * Implementation of route function in Topology.
     */
    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;

    /**
     * Get connection policies
     * Each connection policy is represented as a pair of (src, dest) device ids.
     * For a 4-node mesh, the connection policies are:
     * - if bidirectional: (0,1), (1,2), (2,3), (1,0), (2,1), (3,2)
     *
     * @return list of connection policies
     */
    [[nodiscard]] std::vector<ConnectionPolicy> get_connection_policies() const noexcept override;
};

}  // namespace NetworkAnalyticalCongestionAware
