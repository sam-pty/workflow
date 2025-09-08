/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __MESH_TOPOLOGY_HH__
#define __MESH_TOPOLOGY_HH__

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/system/astraccl/native_collectives/logical_topology/BasicLogicalTopology.hh"

namespace AstraSim {

class MeshTopology : public BasicLogicalTopology {
  public:
    enum class Direction { Clockwise, Anticlockwise };
    enum class Dimension { Local, Vertical, Horizontal, NA };
    int get_num_of_nodes_in_dimension(int dimension) override;
    MeshTopology(Dimension dimension,
                 int id,
                 int total_nodes_in_mesh,
                 int index_in_mesh,
                 int offset);
    MeshTopology(Dimension dimension, int id, std::vector<int> NPUs);
    virtual int get_receiver(int node_id, Direction direction);
    virtual int get_sender(int node_id, Direction direction);
    int get_nodes_in_mesh();
    bool is_enabled();
    Dimension get_dimension();
    int get_index_in_mesh();

  private:
    std::unordered_map<int, int> id_to_index;
    std::unordered_map<int, int> index_to_id;

    std::string name;
    int id;
    int offset;
    int total_nodes_in_mesh;
    int index_in_mesh;
    Dimension dimension;

    virtual int get_receiver_homogeneous(int node_id,
                                         Direction direction,
                                         int offset);
};

}  // namespace AstraSim

#endif /* __MESH_TOPOLOGY_HH__ */
