/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include <unordered_map>
#include <vector>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * Represents a switch translation unit.
 * Translate switch Address to device ID, it is a helper class for MultiDimTopology.
 */
class SwitchTranslationUnit final {
  public:
    /**
     * Constructor.
     *
     * @param npus_count number of npus connected to the switch
     * @param bandwidth bandwidth of link
     * @param latency latency of link
     */
    SwitchTranslationUnit(std::vector<int> npus_count_per_dim, std::vector<bool> is_switch_dim) noexcept;

    /** Translates a switch address to a device ID.
     *
     * @param address The switch address.
     * @return The device ID.
     */
    [[nodiscard]] DeviceId translate_address_to_id(const MultiDimAddress& address) const noexcept;

  private:
    /** Translates a partial address to an offset.
     *
     * @param partial_address The partial address.
     * @param partial_npus_count_per_dim The number of NPUs per dimension for the partial address.
     * @return The offset.
     */
    [[nodiscard]] DeviceId translate_partial_address_to_offset(
        const MultiDimAddress& partial_address, const std::vector<int>& partial_npus_count_per_dim) const noexcept;

    /// Total number of NPUs connected to the switch.
    const int m_total_npus_count;
    /// BasicTopology instances per dimension.
    const std::vector<int> m_npus_count_per_dim;
    /// indicates which dimensions are switches.
    const std::vector<bool> m_is_switch_dim;
    /// map of address length to starting offset of such switches
    std::unordered_map<int, int> m_switch_length_number_mapping;
};

}  // namespace NetworkAnalyticalCongestionAware
