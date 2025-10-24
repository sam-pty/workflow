/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/SwitchTranslationUnit.h"
#include <cassert>
#include <iostream>
#include <numeric>

namespace NetworkAnalyticalCongestionAware {

SwitchTranslationUnit::SwitchTranslationUnit(std::vector<int> npus_count_per_dim,
                                             std::vector<bool> is_switch_dim) noexcept
    : m_total_npus_count{std::accumulate(
          npus_count_per_dim.begin(), npus_count_per_dim.end(), 1, std::multiplies<int>())},
      m_npus_count_per_dim{npus_count_per_dim},
      m_is_switch_dim{is_switch_dim} {
    assert(npus_count_per_dim.size() == is_switch_dim.size());

    // build switch length to starting offset mapping
    int cumulative_offset = 0;
    int dims_count = m_npus_count_per_dim.size();
    for (int dim = 0; dim < dims_count; dim++) {
        if (m_is_switch_dim.at(dim)) {
            // product of npus in all higher dimensions
            int num_switches = std::accumulate(m_npus_count_per_dim.begin() + dim + 1, m_npus_count_per_dim.end(),
                                               static_cast<int>(1), std::multiplies<int>());

            int address_length = dims_count - dim - 1;
            assert(m_switch_length_number_mapping.find(address_length) == m_switch_length_number_mapping.end());
            m_switch_length_number_mapping.insert({address_length, cumulative_offset});
            cumulative_offset += num_switches;
        }
    }
}

[[nodiscard]] DeviceId SwitchTranslationUnit::translate_address_to_id(const MultiDimAddress& address) const noexcept {
    // find which dimension is the switch
    int switch_dim = -1;
    for (int dim = 0; dim < address.size(); dim++) {
        if (address.at(dim) == m_npus_count_per_dim.at(dim)) {
            switch_dim = dim;
            break;
        }
    }
    assert(switch_dim != -1);

    // get the length of rest address
    int left_length = address.size() - switch_dim - 1;
    MultiDimAddress partial_addr(address.begin() + switch_dim + 1, address.end());
    // get offset on that level
    DeviceId offset = translate_partial_address_to_offset(
        partial_addr, std::vector<int>(m_npus_count_per_dim.begin() + switch_dim + 1, m_npus_count_per_dim.end()));

    return m_total_npus_count + m_switch_length_number_mapping.at(left_length) + offset;
}

DeviceId SwitchTranslationUnit::translate_partial_address_to_offset(
    const MultiDimAddress& partial_address, const std::vector<int>& partial_npus_count_per_dim) const noexcept {
    // roughly duplicate of MultiDimTopology::translate_address_back
    DeviceId device_id = 0;
    for (int top_dim = partial_npus_count_per_dim.size() - 1; top_dim >= 0; top_dim--) {
        DeviceId total_npus_in_group =
            std::accumulate(partial_npus_count_per_dim.begin(), partial_npus_count_per_dim.begin() + top_dim,
                            static_cast<DeviceId>(1), std::multiplies<DeviceId>());

        device_id += total_npus_in_group * partial_address.at(top_dim);
    }
    return device_id;
}

};  // namespace NetworkAnalyticalCongestionAware
