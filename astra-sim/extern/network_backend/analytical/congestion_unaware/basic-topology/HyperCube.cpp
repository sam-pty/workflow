/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/HyperCube.h"
#include <bitset>
#include <cassert>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

HyperCube::HyperCube(const int npus_count,
                     const Bandwidth bandwidth,
                     const Latency latency,
                     const bool bidirectional) noexcept
    : bidirectional(bidirectional),
      BasicTopology(npus_count, bandwidth, latency) {
    assert(npus_count > 0);
    assert((npus_count & (npus_count - 1)) == 0);  // must be power of 2
    assert(bandwidth > 0);
    assert(latency >= 0);

    // set the building block type
    basic_topology_type = TopologyBuildingBlock::HyperCube;
}

int HyperCube::compute_hops_count(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(src != dest);

    // For HyperCube topology:
    // Distance = Hamming distance between src and dest
    unsigned int diff = src ^ dest;  // XOR shows differing bits
    int hops = 0;

    while (diff) {
        hops += (diff & 1);
        diff >>= 1;
    }

    return hops;
}
