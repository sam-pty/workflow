/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/MeshTopology.hh"
#include "astra-sim/common/Logging.hh"

#include <cassert>
#include <iostream>

using namespace std;
using namespace AstraSim;

MeshTopology::MeshTopology(Dimension dimension, int id, std::vector<int> NPUs)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::Mesh) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    this->id = id;
    this->total_nodes_in_mesh = NPUs.size();
    this->dimension = dimension;
    this->offset = -1;
    this->index_in_mesh = -1;
    for (int i = 0; i < total_nodes_in_mesh; i++) {
        id_to_index[NPUs[i]] = i;
        index_to_id[i] = NPUs[i];
        if (id == NPUs[i]) {
            index_in_mesh = i;
        }
    }

    LoggerFactory::get_logger("system::topology::MeshTopology")
        ->info("custom mesh, id: {}, dimension: {} total nodes in mesh: {} "
               "index in mesh: {} total nodes in mesh {}",
               id, name, total_nodes_in_mesh, index_in_mesh,
               total_nodes_in_mesh);

    assert(index_in_mesh >= 0);
}
MeshTopology::MeshTopology(Dimension dimension,
                           int id,
                           int total_nodes_in_mesh,
                           int index_in_mesh,
                           int offset)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::Mesh) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    if (id == 0) {
        LoggerFactory::get_logger("system::topology::MeshTopology")
            ->info("mesh of node 0, id: {} dimension: {} total nodes in mesh: "
                   "{} index in mesh: {} offset: {} total nodes in mesh: {}",
                   id, name, total_nodes_in_mesh, index_in_mesh, offset,
                   total_nodes_in_mesh);
    }
    this->id = id;
    this->total_nodes_in_mesh = total_nodes_in_mesh;
    this->index_in_mesh = index_in_mesh;
    this->dimension = dimension;
    this->offset = offset;

    id_to_index[id] = index_in_mesh;
    index_to_id[index_in_mesh] = id;
    int tmp = id;
    for (int i = 0; i < total_nodes_in_mesh - 1; i++) {
        tmp = get_receiver_homogeneous(tmp, MeshTopology::Direction::Clockwise,
                                       offset);
    }
}

int MeshTopology::get_receiver_homogeneous(int node_id,
                                           Direction direction,
                                           int offset) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == MeshTopology::Direction::Clockwise) {
        int receiver = node_id + offset;
        if (index == total_nodes_in_mesh - 1) {
            receiver -= (total_nodes_in_mesh * offset);
            index = 0;
        } else {
            index++;
        }
        if (receiver < 0) {
            LoggerFactory::get_logger("system::topology::MeshTopology")
                ->critical("at dim: {} at id: {} dimension: {} index: {}, node "
                           "id: {}, offset: {}, index_in_mesh {} receiver {}",
                           name, id, name, index, node_id, offset,
                           index_in_mesh, receiver);
        }
        assert(receiver >= 0);
        id_to_index[receiver] = index;
        index_to_id[index] = receiver;
        return receiver;
    } else {
        int receiver = node_id - offset;
        if (index == 0) {
            receiver += (total_nodes_in_mesh * offset);
            index = total_nodes_in_mesh - 1;
        } else {
            index--;
        }
        if (receiver < 0) {
            LoggerFactory::get_logger("system::topology::MeshTopology")
                ->critical("at dim: {} at id: {} dimension: {} index: {}, node "
                           "id: {}, offset: {}, index_in_mesh {} receiver {}",
                           name, id, name, index, node_id, offset,
                           index_in_mesh, receiver);
        }
        assert(receiver >= 0);
        id_to_index[receiver] = index;
        index_to_id[index] = receiver;
        return receiver;
    }
}

int MeshTopology::get_receiver(int node_id, Direction direction) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == MeshTopology::Direction::Clockwise) {
        index++;
        if (index == total_nodes_in_mesh) {
            index = 0;
        }
        return index_to_id[index];
    } else {
        index--;
        if (index < 0) {
            index = total_nodes_in_mesh - 1;
        }
        return index_to_id[index];
    }
}

int MeshTopology::get_sender(int node_id, Direction direction) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == MeshTopology::Direction::Anticlockwise) {
        index++;
        if (index == total_nodes_in_mesh) {
            index = 0;
        }
        return index_to_id[index];
    } else {
        index--;
        if (index < 0) {
            index = total_nodes_in_mesh - 1;
        }
        return index_to_id[index];
    }
}

int MeshTopology::get_index_in_mesh() {
    return index_in_mesh;
}

MeshTopology::Dimension MeshTopology::get_dimension() {
    return dimension;
}

int MeshTopology::get_num_of_nodes_in_dimension(int dimension) {
    return get_nodes_in_mesh();
}

int MeshTopology::get_nodes_in_mesh() {
    return total_nodes_in_mesh;
}

bool MeshTopology::is_enabled() {
    assert(offset > 0);
    int tmp_index = index_in_mesh;
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
