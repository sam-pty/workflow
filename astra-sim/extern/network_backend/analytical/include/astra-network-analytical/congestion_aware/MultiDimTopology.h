/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/BasicTopology.h"
#include "congestion_aware/SwitchTranslationUnit.h"
#include "congestion_aware/Topology.h"

#include <memory>
#include <optional>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * MultiDimTopology implements multi-dimensional network topologies
 * which can be constructed by stacking up multiple BasicTopology instances.
 */
class MultiDimTopology : public Topology {
  public:
    /**
     * Constructor.
     */
    MultiDimTopology() noexcept;

    /**
     * Implementation of route function in Topology.
     */
    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;

    /**
     * Add a dimension to the multi-dimensional topology.
     *
     * @param topology BasicTopology instance to be added.
     */
    void append_dimension(std::unique_ptr<BasicTopology> basic_topology) noexcept;

    /**
     * Make connections for all nodes inter and intra dimensions.
     */
    void make_connections() noexcept;

    /**
     * Initialize all devices in the topology.
     */
    void initialize_all_devices() noexcept;

    /**
     * Build mapping from switch address length to starting offset.
     */
    void build_switch_length_mapping() noexcept;

  private:
    /**
     * Translate the NPU ID into a multi-dimensional address.
     *
     * @param npu_id id of the NPU
     * @return the same NPU in multi-dimensional address representation
     */
    [[nodiscard]] MultiDimAddress translate_address(DeviceId npu_id) const noexcept;

    [[nodiscard]] DeviceId translate_address_back(const MultiDimAddress multi_dim_address) const noexcept;

    /**
     * Given src and dest address in multi-dimensional form,
     * return the dimension where the transfer should happen.
     * i.e., the dimension where the src and dest addresses differ.
     *
     * @param src_address src NPU ID in multi-dimensional form
     * @param dest_address dest NPU ID in multi-dimensional form
     * @return the dimension where the transfer should happen
     */
    [[nodiscard]] int get_dim_to_transfer(const MultiDimAddress& src_address,
                                          const MultiDimAddress& dest_address) const noexcept;

    /** Get the number of devices per each dimension.
     *  For example, if the topology is [2, 8, 4], then this function returns [32, 4, 1].
     *
     * @return number of devices per each dimension
     */
    [[nodiscard]] int get_total_num_devices() const noexcept;

    /**
     * Check if the given address corresponds to a switch device.
     *
     * @param address multi-dimensional address to check
     * @return @c true if the address corresponds to a switch, false otherwise
     */
    [[nodiscard]] bool is_switch(const MultiDimAddress& address) const noexcept;

    /// BasicTopology instances per dimension.
    std::vector<std::unique_ptr<BasicTopology>> m_topology_per_dim;
    /// Switch translation unit for address to device ID translation.
    std::optional<SwitchTranslationUnit> m_switch_translation_unit;
};

}  // namespace NetworkAnalyticalCongestionAware
