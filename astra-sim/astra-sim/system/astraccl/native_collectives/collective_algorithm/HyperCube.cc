/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <cmath>

#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/HyperCube.hh"

#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"

using namespace AstraSim;

HyperCube::HyperCube(ComType type,
                     int id,
                     HyperCubeTopology* hypercube_topology,
                     uint64_t data_size,
                     HyperCubeTopology::Direction direction,
                     InjectionPolicy injection_policy)
    : Algorithm() {
    this->comType = type;
    this->id = id;
    this->logical_topo = hypercube_topology;
    this->data_size = data_size;
    this->direction = direction;
    this->nodes_in_hypercube = hypercube_topology->get_nodes_in_hypercube();
    this->curr_receiver = hypercube_topology->get_receiver(id, direction);
    this->curr_sender = hypercube_topology->get_sender(id, direction);
    this->parallel_reduce = 1;
    this->injection_policy = injection_policy;
    this->total_packets_sent = 0;
    this->total_packets_received = 0;
    this->free_packets = 0;
    this->zero_latency_packets = 0;
    this->non_zero_latency_packets = 0;
    this->toggle = false;
    this->name = Name::HyperCube;
    if (hypercube_topology->get_dimension() ==
        HyperCubeTopology::Dimension::Local) {
        transmition = MemBus::Transmition::Fast;
    } else {
        transmition = MemBus::Transmition::Usual;
    }
    switch (type) {
    case ComType::All_Reduce:
        stream_count = 2 * std::ceil(std::log(nodes_in_hypercube));
        break;
    case ComType::All_to_All:
        this->stream_count =
            ((nodes_in_hypercube - 1) * nodes_in_hypercube) / 2;
        switch (injection_policy) {
        case InjectionPolicy::Aggressive:
            this->parallel_reduce = nodes_in_hypercube - 1;
            break;
        case InjectionPolicy::Normal:
            this->parallel_reduce = 1;
            break;
        default:
            this->parallel_reduce = 1;
            break;
        }
        break;
    default:
        stream_count = std::ceil(std::log(nodes_in_hypercube));
    }
    if (type == ComType::All_to_All || type == ComType::All_Gather) {
        max_count = 0;
    } else {
        max_count = nodes_in_hypercube - 1;
    }
    remained_packets_per_message = 1;
    remained_packets_per_max_count = 1;
    switch (type) {
    case ComType::All_Reduce:
        this->final_data_size = data_size;
        this->msg_size = data_size / nodes_in_hypercube;
        break;
    case ComType::All_Gather:
        this->final_data_size = data_size * nodes_in_hypercube;
        this->msg_size = data_size;
        break;
    case ComType::Reduce_Scatter:
        this->final_data_size = data_size / nodes_in_hypercube;
        this->msg_size = data_size / nodes_in_hypercube;
        break;
    case ComType::All_to_All:
        this->final_data_size = data_size;
        this->msg_size = data_size / nodes_in_hypercube;
        break;
    default:;
    }
}

int HyperCube::get_non_zero_latency_packets() {
    return (nodes_in_hypercube - 1) * parallel_reduce * 1;
}

void HyperCube::run(EventType event, CallData* data) {
    if (event == EventType::General) {
        free_packets += 1;
        ready();
        iteratable();
    } else if (event == EventType::PacketReceived) {
        total_packets_received++;
        insert_packet(nullptr);
    } else if (event == EventType::StreamInit) {
        for (int i = 0; i < parallel_reduce; i++) {
            insert_packet(nullptr);
        }
    }
}

void HyperCube::release_packets() {
    for (auto packet : locked_packets) {
        packet->set_notifier(this);
    }
    if (NPU_to_MA == true) {
        (new PacketBundle(stream->owner, stream, locked_packets, processed,
                          send_back, msg_size, transmition))
            ->send_to_MA();
    } else {
        (new PacketBundle(stream->owner, stream, locked_packets, processed,
                          send_back, msg_size, transmition))
            ->send_to_NPU();
    }
    locked_packets.clear();
}

void HyperCube::process_stream_count() {
    if (remained_packets_per_message > 0) {
        remained_packets_per_message--;
    }
    if (id == 0) {
    }
    if (remained_packets_per_message == 0 && stream_count > 0) {
        stream_count--;
        if (stream_count > 0) {
            remained_packets_per_message = 1;
        }
    }
    if (remained_packets_per_message == 0 && stream_count == 0 &&
        stream->state != StreamState::Dead) {
        stream->changeState(StreamState::Zombie);
    }
}

void HyperCube::process_max_count() {
    if (remained_packets_per_max_count > 0) {
        remained_packets_per_max_count--;
    }
    if (remained_packets_per_max_count == 0) {
        max_count--;
        release_packets();
        remained_packets_per_max_count = 1;
    }
}

void HyperCube::reduce() {
    process_stream_count();
    packets.pop_front();
    free_packets--;
    total_packets_sent++;
}

bool HyperCube::iteratable() {
    if (stream_count == 0 &&
        free_packets == (parallel_reduce * 1)) {  // && not_delivered==0
        exit();
        return false;
    }
    return true;
}

void HyperCube::insert_packet(Callable* sender) {
    if (zero_latency_packets == 0 && non_zero_latency_packets == 0) {
        zero_latency_packets = parallel_reduce * 1;
        non_zero_latency_packets =
            get_non_zero_latency_packets();  //(nodes_in_hypercube-1)*parallel_reduce*1;
        toggle = !toggle;
    }
    if (zero_latency_packets > 0) {
        packets.push_back(MyPacket(
            stream->current_queue_id, curr_sender,
            curr_receiver));  // vnet Must be changed for alltoall topology
        packets.back().sender = sender;
        locked_packets.push_back(&packets.back());
        processed = false;
        send_back = false;
        NPU_to_MA = true;
        process_max_count();
        zero_latency_packets--;
        return;
    } else if (non_zero_latency_packets > 0) {
        packets.push_back(MyPacket(
            stream->current_queue_id, curr_sender,
            curr_receiver));  // vnet Must be changed for alltoall topology
        packets.back().sender = sender;
        locked_packets.push_back(&packets.back());
        if (comType == ComType::Reduce_Scatter ||
            (comType == ComType::All_Reduce && toggle)) {
            processed = true;
        } else {
            processed = false;
        }
        if (non_zero_latency_packets <= parallel_reduce * 1) {
            send_back = false;
        } else {
            send_back = true;
        }
        NPU_to_MA = false;
        process_max_count();
        non_zero_latency_packets--;
        return;
    }
    Sys::sys_panic("should not inject nothing!");
}

bool HyperCube::ready() {
    if (stream->state == StreamState::Created ||
        stream->state == StreamState::Ready) {
        stream->changeState(StreamState::Executing);
    }
    if (packets.size() == 0 || stream_count == 0 || free_packets == 0) {
        return false;
    }
    MyPacket packet = packets.front();
    sim_request snd_req;
    snd_req.srcRank = id;
    snd_req.dstRank = packet.preferred_dest;
    snd_req.tag = stream->stream_id;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    stream->owner->front_end_sim_send(
        0, Sys::dummy_data, msg_size, UINT8, packet.preferred_dest,
        stream->stream_id, &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent,
        nullptr);  // stream_id+(packet.preferred_dest*50)
    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
        stream, stream->owner->id, EventType::PacketReceived,
        packet.preferred_vnet, packet.stream_id);
    stream->owner->front_end_sim_recv(
        0, Sys::dummy_data, msg_size, UINT8, packet.preferred_src,
        stream->stream_id, &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent,
        ehd);  // stream_id+(owner->id*50)
    reduce();
    return true;
}

void HyperCube::exit() {
    if (packets.size() != 0) {
        packets.clear();
    }
    if (locked_packets.size() != 0) {
        locked_packets.clear();
    }
    stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
    return;
}
