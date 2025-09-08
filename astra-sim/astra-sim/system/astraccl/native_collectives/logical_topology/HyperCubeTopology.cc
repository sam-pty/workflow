/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/HyperCubeTopology.hh"
#include "astra-sim/common/Logging.hh"

#include <cassert>
#include <iostream>

using namespace std;
using namespace AstraSim;

HyperCubeTopology::HyperCubeTopology(Dimension dimension,
                                     int id,
                                     std::vector<int> NPUs)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::HyperCube) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    this->id = id;
    this->total_nodes_in_hypercube = NPUs.size();
    this->dimension = dimension;
    this->offset = -1;
    this->index_in_hypercube = -1;
    for (int i = 0; i < total_nodes_in_hypercube; i++) {
        id_to_index[NPUs[i]] = i;
        index_to_id[i] = NPUs[i];
        if (id == NPUs[i]) {
            index_in_hypercube = i;
        }
    }

    LoggerFactory::get_logger("system::topology::HyperCubeTopology")
        ->info("custom hypercube, id: {}, dimension: {} total nodes in "
               "hypercube: {} "
               "index in hypercube: {} total nodes in hypercube {}",
               id, name, total_nodes_in_hypercube, index_in_hypercube,
               total_nodes_in_hypercube);

    assert(index_in_hypercube >= 0);
}
HyperCubeTopology::HyperCubeTopology(Dimension dimension,
                                     int id,
                                     int total_nodes_in_hypercube,
                                     int index_in_hypercube,
                                     int offset)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::HyperCube) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    if (id == 0) {
        LoggerFactory::get_logger("system::topology::HyperCubeTopology")
            ->info("hypercube of node 0, id: {} dimension: {} total nodes in "
                   "hypercube: "
                   "{} index in hypercube: {} offset: {} total nodes in "
                   "hypercube: {}",
                   id, name, total_nodes_in_hypercube, index_in_hypercube,
                   offset, total_nodes_in_hypercube);
    }
    this->id = id;
    this->total_nodes_in_hypercube = total_nodes_in_hypercube;
    this->index_in_hypercube = index_in_hypercube;
    this->dimension = dimension;
    this->offset = offset;

    id_to_index[id] = index_in_hypercube;
    index_to_id[index_in_hypercube] = id;
    int tmp = id;
    for (int i = 0; i < total_nodes_in_hypercube - 1; i++) {
        tmp = get_receiver_homogeneous(
            tmp, HyperCubeTopology::Direction::Clockwise, offset);
    }
}

int HyperCubeTopology::get_receiver_homogeneous(int node_id,
                                                Direction direction,
                                                int offset) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == HyperCubeTopology::Direction::Clockwise) {
        int receiver = node_id + offset;
        if (index == total_nodes_in_hypercube - 1) {
            receiver -= (total_nodes_in_hypercube * offset);
            index = 0;
        } else {
            index++;
        }
        if (receiver < 0) {
            LoggerFactory::get_logger("system::topology::HyperCubeTopology")
                ->critical(
                    "at dim: {} at id: {} dimension: {} index: {}, node "
                    "id: {}, offset: {}, index_in_hypercube {} receiver {}",
                    name, id, name, index, node_id, offset, index_in_hypercube,
                    receiver);
        }
        assert(receiver >= 0);
        id_to_index[receiver] = index;
        index_to_id[index] = receiver;
        return receiver;
    } else {
        int receiver = node_id - offset;
        if (index == 0) {
            receiver += (total_nodes_in_hypercube * offset);
            index = total_nodes_in_hypercube - 1;
        } else {
            index--;
        }
        if (receiver < 0) {
            LoggerFactory::get_logger("system::topology::HyperCubeTopology")
                ->critical(
                    "at dim: {} at id: {} dimension: {} index: {}, node "
                    "id: {}, offset: {}, index_in_hypercube {} receiver {}",
                    name, id, name, index, node_id, offset, index_in_hypercube,
                    receiver);
        }
        assert(receiver >= 0);
        id_to_index[receiver] = index;
        index_to_id[index] = receiver;
        return receiver;
    }
}

int HyperCubeTopology::get_receiver(int node_id, Direction direction) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == HyperCubeTopology::Direction::Clockwise) {
        index++;
        if (index == total_nodes_in_hypercube) {
            index = 0;
        }
        return index_to_id[index];
    } else {
        index--;
        if (index < 0) {
            index = total_nodes_in_hypercube - 1;
        }
        return index_to_id[index];
    }
}

int HyperCubeTopology::get_sender(int node_id, Direction direction) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == HyperCubeTopology::Direction::Anticlockwise) {
        index++;
        if (index == total_nodes_in_hypercube) {
            index = 0;
        }
        return index_to_id[index];
    } else {
        index--;
        if (index < 0) {
            index = total_nodes_in_hypercube - 1;
        }
        return index_to_id[index];
    }
}

int HyperCubeTopology::get_index_in_hypercube() {
    return index_in_hypercube;
}

HyperCubeTopology::Dimension HyperCubeTopology::get_dimension() {
    return dimension;
}

int HyperCubeTopology::get_num_of_nodes_in_dimension(int dimension) {
    return get_nodes_in_hypercube();
}

int HyperCubeTopology::get_nodes_in_hypercube() {
    return total_nodes_in_hypercube;
}

bool HyperCubeTopology::is_enabled() {
    assert(offset > 0);
    int tmp_index = index_in_hypercube;
    int tmp_id = id;
    while (tmp_index > 0) {
        tmp_index--;
        tmp_id -= offset;
    }
    if (tmp_id == 0) {
        return true;
    }
    return false;
}
