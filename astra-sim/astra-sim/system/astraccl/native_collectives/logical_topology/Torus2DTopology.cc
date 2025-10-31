/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/Torus2DTopology.hh"
#include "astra-sim/common/Logging.hh"

#include <cassert>
#include <iostream>

using namespace std;
using namespace AstraSim;

Torus2DTopology::Torus2DTopology(Dimension dimension, int id, std::vector<int> NPUs)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::Torus2D) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    this->id = id;
    this->total_nodes_in_ring = NPUs.size();
    this->dimension = dimension;
    this->offset = -1;
    this->index_in_ring = -1;
    
    // Initialize dims
    if (dimension == Dimension::Local) {
        dims = {total_nodes_in_ring};
    } else if (dimension == Dimension::Horizontal || dimension == Dimension::Vertical) {
        int dim_size = static_cast<int>(std::sqrt(total_nodes_in_ring));
        dims = {dim_size, dim_size};
    } else {
        dims = {total_nodes_in_ring};
    }

    for (int i = 0; i < total_nodes_in_ring; i++) {
        id_to_index[NPUs[i]] = i;
        index_to_id[i] = NPUs[i];
        if (id == NPUs[i]) {
            index_in_ring = i;
        }
    }

    LoggerFactory::get_logger("system::topology::Torus2DTopology")
        ->info("custom ring, id: {}, dimension: {} total nodes in ring: {} "
               "index in ring: {} total nodes in ring {}",
               id, name, total_nodes_in_ring, index_in_ring,
               total_nodes_in_ring);

    assert(index_in_ring >= 0);
}

Torus2DTopology::Torus2DTopology(Dimension dimension,
                           int id,
                           int total_nodes_in_ring,
                           int index_in_ring,
                           int offset)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::Ring) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    if (id == 0) {
        LoggerFactory::get_logger("system::topology::Torus2DTopology")
            ->info("ring of node 0, id: {} dimension: {} total nodes in ring: "
                   "{} index in ring: {} offset: {} total nodes in ring: {}",
                   id, name, total_nodes_in_ring, index_in_ring, offset,
                   total_nodes_in_ring);
    }
    this->id = id;
    this->total_nodes_in_ring = total_nodes_in_ring;
    this->index_in_ring = index_in_ring;
    this->dimension = dimension;
    this->offset = offset;

    // Initialize dims
    if (dimension == Dimension::Local) {
        dims = {total_nodes_in_ring};
    } else if (dimension == Dimension::Horizontal || dimension == Dimension::Vertical) {
        int dim_size = static_cast<int>(std::sqrt(total_nodes_in_ring));
        dims = {dim_size, dim_size};
    } else {
        dims = {total_nodes_in_ring};
    }

    id_to_index[id] = index_in_ring;
    index_to_id[index_in_ring] = id;
    int tmp = id;
    for (int i = 0; i < total_nodes_in_ring - 1; i++) {
        tmp = get_receiver_homogeneous(tmp, Torus2DTopology::Direction::Clockwise,
                                       offset);
    }
}


int Torus2DTopology::get_receiver_homogeneous(int node_id,
                                           Direction direction,
                                           int offset) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == Torus2DTopology::Direction::Clockwise) {
        int receiver = node_id + offset;
        if (index == total_nodes_in_ring - 1) {
            receiver -= (total_nodes_in_ring * offset);
            index = 0;
        } else {
            index++;
        }
        if (receiver < 0) {
            LoggerFactory::get_logger("system::topology::Torus2DTopology")
                ->critical("at dim: {} at id: {} dimension: {} index: {}, node "
                           "id: {}, offset: {}, index_in_ring {} receiver {}",
                           name, id, name, index, node_id, offset,
                           index_in_ring, receiver);
        }
        assert(receiver >= 0);
        id_to_index[receiver] = index;
        index_to_id[index] = receiver;
        return receiver;
    } else {
        int receiver = node_id - offset;
        if (index == 0) {
            receiver += (total_nodes_in_ring * offset);
            index = total_nodes_in_ring - 1;
        } else {
            index--;
        }
        if (receiver < 0) {
            LoggerFactory::get_logger("system::topology::Torus2DTopology")
                ->critical("at dim: {} at id: {} dimension: {} index: {}, node "
                           "id: {}, offset: {}, index_in_ring {} receiver {}",
                           name, id, name, index, node_id, offset,
                           index_in_ring, receiver);
        }
        assert(receiver >= 0);
        id_to_index[receiver] = index;
        index_to_id[index] = receiver;
        return receiver;
    }
}



std::vector<int> Torus2DTopology::get_receivers(int node_id, Torus2DTopology::Direction direction) const {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index.at(node_id);

    std::vector<int> receivers;

    // 1D ring fallback
    if (dims.empty() || dims.size() == 1) {
        int n = total_nodes_in_ring;
        if (n <= 1) return receivers;
        if (direction == Torus2DTopology::Direction::Clockwise) {
            int cw_index = (index + 1) % n;
            receivers.push_back(index_to_id.at(cw_index));
        } else {
            int ccw_index = (index - 1 + n) % n;
            receivers.push_back(index_to_id.at(ccw_index));
        }
        return receivers;
    }

    // multi-dimensional case
    const int num_dims = static_cast<int>(dims.size());
    std::vector<int> coords(num_dims);

    int tmp = index;
    for (int d = num_dims - 1; d >= 0; --d) {
        coords[d] = tmp % dims[d];
        tmp /= dims[d];
    }

    if (direction == Torus2DTopology::Direction::Clockwise) {
        // Clockwise => +1 along each dimension
        for (int d = 0; d < num_dims; ++d) {
            std::vector<int> new_coords = coords;
            new_coords[d] = (new_coords[d] + 1) % dims[d];
            int new_index = 0;
            for (int k = 0; k < num_dims; ++k) {
                new_index = new_index * dims[k] + new_coords[k];
            }
            receivers.push_back(index_to_id.at(new_index));
        }
    } else {
        // Anticlockwise => -1 along each dimension
        for (int d = 0; d < num_dims; ++d) {
            std::vector<int> new_coords = coords;
            new_coords[d] = (new_coords[d] - 1 + dims[d]) % dims[d];
            int new_index = 0;
            for (int k = 0; k < num_dims; ++k) {
                new_index = new_index * dims[k] + new_coords[k];
            }
            receivers.push_back(index_to_id.at(new_index));
        }
    }

    return receivers;
}



std::vector<int> Torus2DTopology::get_senders(int node_id, Torus2DTopology::Direction direction) const {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index.at(node_id);

    std::vector<int> receivers;

    // 1D ring fallback
    if (dims.empty() || dims.size() == 1) {
        int n = total_nodes_in_ring;
        if (n <= 1) return receivers;
        if (direction == Torus2DTopology::Direction::Anticlockwise) {
            int cw_index = (index + 1) % n;
            receivers.push_back(index_to_id.at(cw_index));
        } else {
            int ccw_index = (index - 1 + n) % n;
            receivers.push_back(index_to_id.at(ccw_index));
        }
        return receivers;
    }

    // multi-dimensional case
    const int num_dims = static_cast<int>(dims.size());
    std::vector<int> coords(num_dims);

    int tmp = index;
    for (int d = num_dims - 1; d >= 0; --d) {
        coords[d] = tmp % dims[d];
        tmp /= dims[d];
    }

    if (direction == Torus2DTopology::Direction::Anticlockwise) {
        // Clockwise => +1 along each dimension
        for (int d = 0; d < num_dims; ++d) {
            std::vector<int> new_coords = coords;
            new_coords[d] = (new_coords[d] + 1) % dims[d];
            int new_index = 0;
            for (int k = 0; k < num_dims; ++k) {
                new_index = new_index * dims[k] + new_coords[k];
            }
            receivers.push_back(index_to_id.at(new_index));
        }
    } else {
        // Anticlockwise => -1 along each dimension
        for (int d = 0; d < num_dims; ++d) {
            std::vector<int> new_coords = coords;
            new_coords[d] = (new_coords[d] - 1 + dims[d]) % dims[d];
            int new_index = 0;
            for (int k = 0; k < num_dims; ++k) {
                new_index = new_index * dims[k] + new_coords[k];
            }
            receivers.push_back(index_to_id.at(new_index));
        }
    }

    return receivers;
}


int Torus2DTopology::get_index_in_ring() {
    return index_in_ring;
}

Torus2DTopology::Dimension Torus2DTopology::get_dimension() {
    return dimension;
}

int Torus2DTopology::get_num_of_nodes_in_dimension(int dimension) {
    return get_nodes_in_ring();
}

int Torus2DTopology::get_nodes_in_ring() {
    return total_nodes_in_ring;
}

bool Torus2DTopology::is_enabled() {
    assert(offset > 0);
    int tmp_index = index_in_ring;
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
