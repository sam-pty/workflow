/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __TORUS2D_TOPOLOGY_HH__
#define __TORUS2D_TOPOLOGY_HH__

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/system/astraccl/native_collectives/logical_topology/BasicLogicalTopology.hh"

namespace AstraSim {

class Torus2DTopology : public BasicLogicalTopology {
  public:
    enum class Direction { Clockwise, Anticlockwise };
    enum class Dimension { Local, Vertical, Horizontal, NA };
    int get_num_of_nodes_in_dimension(int dimension) override;
    Torus2DTopology(Dimension dimension,
                 int id,
                 int total_nodes_in_ring,
                 int index_in_ring,
                 int offset);
    Torus2DTopology(Dimension dimension, int id, std::vector<int> NPUs);
    virtual std::vector<int> get_receivers(int node_id, Direction direction) const;
    virtual std::vector<int> get_senders(int node_id, Direction direction) const;
    int get_nodes_in_ring();
    bool is_enabled();
    Dimension get_dimension();
    int get_index_in_ring();

  private:
    std::unordered_map<int, int> id_to_index;
    std::unordered_map<int, int> index_to_id;

    std::string name;
    int id;
    int offset;
    int total_nodes_in_ring;
    int index_in_ring;
    

    Dimension dimension;
    
    std::vector<int> dims;  // Sizes per dimension, 
    int num_dimensions;           // = dims.size()


    virtual int get_receiver_homogeneous(int node_id,
                                         Direction direction,
                                         int offset);
};

}  // namespace AstraSim

#endif /* __TORUS2D_TOPOLOGY_HH__ */
