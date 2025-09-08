/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_unaware/BasicTopology.h"

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionUnaware {

/**
 * Implements a hypercube topology.
 *
 * HyperCube(8) example:
 * @todo need to change
 * 0 - 1 - 2 - 3
 * |           |
 * 7 - 6 - 5 - 4
 *
 * If hypercube is uni-directional, then each chunk can flow through:
 * 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0
 *
 * If the hypercube is bi-directional, then each chunk can flow through:
 * 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0
 * 0 <- 1 <- 2 <- 3 <- 4 <- 5 <- 6 <- 7 <- 0
 */
class HyperCube final : public BasicTopology {
  public:
    /**
     * Constructor
     *
     * @param npus_count number of NPUs in the HyperCube
     * @param bandwidth bandwidth of each link
     * @param latency latency of each link
     * @param bidirectional whether the hypercube is bidirectional, defaults to true
     */
    HyperCube(int npus_count, Bandwidth bandwidth, Latency latency, bool bidirectional = true) noexcept;

  private:
    /**
     * Implements the compute_hops_count method of BasicTopology.
     */
    [[nodiscard]] int compute_hops_count(DeviceId src, DeviceId dest) const noexcept override;

    /// true if the hypercube is bidirectional, false otherwise
    bool bidirectional;
};

}  // namespace NetworkAnalyticalCongestionUnaware
