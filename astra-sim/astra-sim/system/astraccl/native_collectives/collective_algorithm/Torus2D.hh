/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __TORUS2D_HH__
#define __TORUS2D_HH__

#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/MyPacket.hh"
#include "astra-sim/system/astraccl/Algorithm.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/Torus2DTopology.hh"

namespace AstraSim {

class Torus2D : public Algorithm {
  public:
    Torus2D(ComType type,
         int id,
         Torus2DTopology* ring_topology,
         uint64_t data_size,
         Torus2DTopology::Direction direction,
         InjectionPolicy injection_policy);
    virtual void run(EventType event, CallData* data);
    void process_stream_count();
    void release_packets();
    virtual void process_max_count();
    void reduce();
    bool iteratable();
    virtual int get_non_zero_latency_packets();
    void insert_packet(Callable* sender);
    bool ready();
    void exit();

    Torus2DTopology::Direction dimension;
    Torus2DTopology::Direction direction;
    MemBus::Transmition transmition;
    int zero_latency_packets;
    int non_zero_latency_packets;
    int id;
    std::vector<int> curr_receivers;
    std::vector<int> curr_senders;
    int nodes_in_dim;
    int stream_count;
    int max_count;
    int remained_packets_per_max_count;
    int remained_packets_per_message;
    int parallel_reduce;
    InjectionPolicy injection_policy;
    std::list<MyPacket> packets;
    bool toggle;
    long free_packets;
    long total_packets_sent;
    long total_packets_received;
    uint64_t msg_size;
    std::list<MyPacket*> locked_packets;
    bool processed;
    bool send_back;
    bool NPU_to_MA;
    bool m_bidirectional;
};

}  // namespace AstraSim

#endif /* __TORUS2D_HH__ */
