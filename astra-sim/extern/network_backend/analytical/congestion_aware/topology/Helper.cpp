/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Helper.h"
#include "congestion_aware/BinaryTree.h"
#include "congestion_aware/DoubleBinaryTree.h"
#include "congestion_aware/FullyConnected.h"
#include "congestion_aware/Mesh.h"
#include "congestion_aware/MultiDimTopology.h"
#include "congestion_aware/Ring.h"
#include "congestion_aware/Torus2D.h"
#include "congestion_aware/Mesh2D.h"
#include "congestion_aware/KingMesh2D.h"
#include "congestion_aware/Switch.h"
#include <cstdlib>
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

std::shared_ptr<Topology> NetworkAnalyticalCongestionAware::construct_topology(
    const NetworkParser& network_parser) noexcept {
    // get network_parser info
    const auto dims_count = network_parser.get_dims_count();
    const auto topologies_per_dim = network_parser.get_topologies_per_dim();
    const auto npus_counts_per_dim = network_parser.get_npus_counts_per_dim();
    const auto bandwidths_per_dim = network_parser.get_bandwidths_per_dim();
    const auto latencies_per_dim = network_parser.get_latencies_per_dim();

    // if single dim, create basic-topology
    if (dims_count == 1) {
        // retrieve basic basic-topology info
        const auto topology_type = topologies_per_dim[0];
        const auto npus_count = npus_counts_per_dim[0];
        const auto bandwidth = bandwidths_per_dim[0];
        const auto latency = latencies_per_dim[0];

        switch (topology_type) {
        case TopologyBuildingBlock::Ring:
            return std::make_shared<Ring>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::Switch:
            return std::make_shared<Switch>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::FullyConnected:
            return std::make_shared<FullyConnected>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::BinaryTree:
            return std::make_shared<BinaryTree>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::DoubleBinaryTree:
            return std::make_shared<DoubleBinaryTree>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::Mesh:
            return std::make_shared<Mesh>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::Torus2D:
            return std::make_shared<Torus2D>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::Mesh2D:
            return std::make_shared<Mesh2D>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::KingMesh2D:
            return std::make_shared<KingMesh2D>(npus_count, bandwidth, latency);
        default:
            // shouldn't reaach here
            std::cerr << "[Error] (network/analytical/congestion_aware) "
                      << "not supported basic-topology" << std::endl;
            std::exit(-1);
        }
    } else {  // otherwise, create multi-dim basic-topology
        const auto multi_dim_topology = std::make_shared<MultiDimTopology>();

        // create and append dims
        for (auto dim = 0; dim < dims_count; dim++) {
            // retrieve info
            const auto topology_type = topologies_per_dim[dim];
            const auto npus_count = npus_counts_per_dim[dim];
            const auto bandwidth = bandwidths_per_dim[dim];
            const auto latency = latencies_per_dim[dim];
            const bool is_multi_dim = true;

            // create a network dim
            std::unique_ptr<BasicTopology> dim_topology;
            switch (topology_type) {
            case TopologyBuildingBlock::Ring:
                dim_topology =
                    std::make_unique<Ring>(npus_count, bandwidth, latency, /* bidirectional = */ false, is_multi_dim);
                break;
            case TopologyBuildingBlock::Switch:
                dim_topology = std::make_unique<Switch>(npus_count, bandwidth, latency, is_multi_dim);
                break;
            case TopologyBuildingBlock::FullyConnected:
                dim_topology = std::make_unique<FullyConnected>(npus_count, bandwidth, latency, is_multi_dim);
                break;
            case TopologyBuildingBlock::BinaryTree:
                dim_topology = std::make_unique<BinaryTree>(npus_count, bandwidth, latency, is_multi_dim);
                break;
            case TopologyBuildingBlock::DoubleBinaryTree:
                dim_topology = std::make_unique<DoubleBinaryTree>(npus_count, bandwidth, latency, is_multi_dim);
                break;
            case TopologyBuildingBlock::Mesh:
                dim_topology = std::make_unique<Mesh>(npus_count, bandwidth, latency, is_multi_dim);
                break;
            case TopologyBuildingBlock::Torus2D:
                dim_topology = std::make_unique<Torus2D>(npus_count, bandwidth, latency, is_multi_dim);
                break;
            case TopologyBuildingBlock::Mesh2D:
                dim_topology = std::make_unique<Mesh2D>(npus_count, bandwidth, latency, is_multi_dim);
                break;

            default:
                // shouldn't reach here
                std::cerr << "[Error] (network/analytical/congestion_aware)"
                          << "Not supported multi-topology" << std::endl;
                std::exit(-1);
            }

            // append network dimension
            multi_dim_topology->append_dimension(std::move(dim_topology));
        }

        multi_dim_topology->initialize_all_devices();
        multi_dim_topology->build_switch_length_mapping();
        multi_dim_topology->make_connections();

        // return created multi-dimensional topology
        return multi_dim_topology;
    }
}

std::vector<std::pair<MultiDimAddress, MultiDimAddress>> NetworkAnalyticalCongestionAware::generateAddressPairs(
    const MultiDimAddress& upper, const ConnectionPolicy& policy, int dim) noexcept {
    std::vector<std::pair<MultiDimAddress, MultiDimAddress>> result;
    MultiDimAddress current(upper.size(), 0);
    generateFreeComb(upper, dim, policy, current, 0, result);
    return result;
}

void NetworkAnalyticalCongestionAware::generateFreeComb(
    const MultiDimAddress& upper,
    int dim,
    const ConnectionPolicy& policy,
    MultiDimAddress& current,
    int index,
    std::vector<std::pair<MultiDimAddress, MultiDimAddress>>& result) noexcept {
    if (index == upper.size()) {
        // Reached the end, save the pair
        result.push_back({current, current});
        result.back().first[dim] = policy.src;
        result.back().second[dim] = policy.dst;
        return;
    }

    if (index == dim) {
        // Skip fixed dimension in recursion
        generateFreeComb(upper, dim, policy, current, index + 1, result);
    } else {
        for (int i = 0; i < upper[index]; ++i) {
            current[index] = i;
            generateFreeComb(upper, dim, policy, current, index + 1, result);
        }
    }
}
