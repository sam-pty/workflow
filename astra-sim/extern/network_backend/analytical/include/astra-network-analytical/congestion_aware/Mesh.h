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
 * Implements a mesh topology.
 *
 * Mesh(8) example:
 * 0 - 1 - 2 - 3
 * |   |   |   |
 * 7 - 6 - 5 - 4
 *
 * Therefore, the number of NPUs and devices are both 8.
 *
 * If mesh is uni-directional, then each chunk can flow through:
 * 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0
 *
 * If the mesh is bi-directional, then each chunk can flow through:
 * 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0
 * 0 <- 1 <- 2 <- 3 <- 4 <- 5 <- 6 <- 7 <- 0
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
    Mesh(int npus_count, Bandwidth bandwidth, Latency latency) noexcept;

    /**
     * Implementation of route function in Topology.
     */
    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;
};

}  // namespace NetworkAnalyticalCongestionAware
