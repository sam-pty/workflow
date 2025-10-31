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
 * Implements a 2DTorus topology.
 *
 * 2DTorus(8) example:
 *    ________________
 *   |_0 - 1 - 2 - 3_|
 *     |   |   |   | 
 *    _7 - 6 - 5 - 4_
 *   |_______________|
 *
 * The number of NPUs and devices are both 8.
 *
 * If ring is uni-directional, then each chunk can flow through:
 * 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0
 *
 * If the ring is bi-directional, then each chunk can flow through:
 * 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 0
 * 0 <- 1 <- 2 <- 3 <- 4 <- 5 <- 6 <- 7 <- 0
 */
class Torus2D final : public BasicTopology {
  public:
    /**
     * Constructor.
     *
     * @param npus_count number of npus in a ring
     * @param bandwidth bandwidth of link
     * @param latency latency of link
     * @param bidirectional true if ring is bidirectional, false otherwise
     */
    Torus2D(int npus_count,
         Bandwidth bandwidth,
         Latency latency,
         bool bidirectional = true,
         bool is_multi_dim = false) noexcept;

    /**
     * Implementation of route function in Topology.
     */
    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;

    /**
     * Get connection policies of the ring topology.
     * Each connection policy is represented as a pair of (src, dest) device ids.
     * For a 4-node ring, the connection policies are:
     * - if bidirectional: (0,1), (1,2), (2,3), (3,0), (1,0), (2,1), (3,2), (0,3)
     *
     * @return list of connection policies
     */
    [[nodiscard]] std::vector<ConnectionPolicy> get_connection_policies() const noexcept override;

  private:

    bool is_faulty(int src, int dst) const;
    /// true if the ring is bidirectional, false otherwise
    bool bidirectional;

    std::vector<std::pair<int, int> > faulty_links;


};

}  // namespace NetworkAnalyticalCongestionAware
