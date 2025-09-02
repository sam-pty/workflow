// /******************************************************************************
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// *******************************************************************************/

// // NOTE: This is a prototype implementation of a 2D mesh all-reduce
// // algorithm modeled after the style of the existing Ring / DoubleBinaryTree
// // algorithms in Astra-Sim. It intentionally keeps the protocol simple and
// // readable: a two-phase reduction (rows -> columns) followed by the reverse
// // broadcast (columns -> rows). You will likely want to tune message
// // bundling, pipelining and edge cases for performance.

// #include "astra-sim/system/astraccl/native_collectives/collective_algorithm/Mesh2DAllReduce.hh"

// #include "astra-sim/system/PacketBundle.hh"
// #include "astra-sim/system/RecvPacketEventHandlerData.hh"

// using namespace AstraSim;

// // -----------------------------------------------------------------------------
// // Implementation notes / assumptions
// // - MeshTopology provides: get_coords(id) -> pair<int x,int y>,
// //   get_width(), get_height(), get_left(id), get_right(id), get_up(id), get_down(id)
// //   and get_node_id(x,y) or similar helpers. If your MeshTopology API differs
// //   adapt the calls below.
// // - We use a simple linear chain per row (reduce toward x==0) and linear chain
// //   per column (reduce toward y==0). At each receive we create a PacketBundle
// //   with `processed=true` to indicate reduction of local + received data.
// // - The code follows the `EventType` driven style used by other algorithms.
// // - This file is a starting point: for production you may want to add
// //   pipelining, splitting of messages into msg_size chunks, and robust
// //   handling for odd sizes/partial receives.
// // -----------------------------------------------------------------------------

// Mesh2DAllReduce::Mesh2DAllReduce(int id, MeshTopology* mesh, uint64_t data_size)
//     : Algorithm() {
//     this->id = id;
//     this->logical_topo = mesh;
//     this->data_size = data_size;
//     this->comType = ComType::All_Reduce;
//     this->name = Name::Mesh2D;

//     // topology helpers
//     std::pair<int, int> coords = mesh->get_coords(id);
//     this->x = coords.first;
//     this->y = coords.second;
//     this->width = mesh->get_width();
//     this->height = mesh->get_height();

//     // neighbor ids (-1 if none)
//     this->left = mesh->get_left(id);
//     this->right = mesh->get_right(id);
//     this->up = mesh->get_up(id);
//     this->down = mesh->get_down(id);

//     // protocol state
//     this->state = State::Begin;
//     this->row_reduced = false;
//     this->col_reduced = false;
//     this->received_from_row_child = false;
//     this->received_from_col_child = false;

//     // message sizing - we keep whole-message semantics for now
//     this->msg_size = data_size;
// }

// void Mesh2DAllReduce::run(EventType event, CallData* data) {
//     // For clarity, we annotate the main phases:
//     // Phase A: Row reduction -> all data in each row is reduced to column 0
//     // Phase B: Column reduction -> column-0 nodes reduce down to (0,0)
//     // Phase C: Broadcast reverse (column 0 -> rows, then row 0 -> all nodes)

//     // Begin state: kick off row reduction listeners / sends depending on x
//     if (state == State::Begin) {
//         // If we have a right neighbor, we expect to receive that neighbor's
//         // partial sum first (chain toward left). We always post a recv for
//         // the right neighbor if it exists.
//         if (right >= 0) {
//             sim_request rcv_req;
//             rcv_req.vnet = this->stream->current_queue_id;
//             RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
//                 stream, stream->owner->id, EventType::PacketReceived,
//                 stream->current_queue_id, stream->stream_id);
//             stream->owner->front_end_sim_recv(
//                 0, Sys::dummy_data, msg_size, UINT8, right, stream->stream_id,
//                 &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE, &Sys::handleEvent,
//                 ehd);
//         }

//         // If we're not the row root (x > 0) we will later send the reduced row
//         // value to the left neighbor. If we are row root (x==0) we wait until
//         // right child arrives (if any) and then proceed to column phase.
//         state = State::WaitingForRowChild;
//         return;
//     }

//     // --- Row reduction ---
//     if (state == State::WaitingForRowChild && event == EventType::PacketReceived) {
//         // Received partial from right child (or multiple children in wider
//         // topologies if you extend chaining). Combine locally (request NPU to
//         // reduce): we signal processed=true so PacketBundle performs reduction
//         // of local + received buffer.
//         (new PacketBundle(stream->owner, stream, true, false, msg_size,
//                           MemBus::Transmition::Usual))
//             ->send_to_NPU();

//         // If we are not the row root, forward the (reduced) value to our left
//         // neighbor by posting a send to the left neighbor and registering a
//         // receive from left for the broadcast later.
//         if (left >= 0) {
//             // send to left
//             sim_request snd_req;
//             snd_req.srcRank = stream->owner->id;
//             snd_req.dstRank = left;
//             snd_req.tag = stream->stream_id;
//             snd_req.reqType = UINT8;
//             snd_req.vnet = this->stream->current_queue_id;
//             stream->owner->front_end_sim_send(
//                 0, Sys::dummy_data, msg_size, UINT8, left, stream->stream_id,
//                 &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE, &Sys::handleEvent,
//                 nullptr);

//             // register recv from left for the later broadcast
//             sim_request rcv_req;
//             rcv_req.vnet = this->stream->current_queue_id;
//             RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
//                 stream, stream->owner->id, EventType::PacketReceived,
//                 stream->current_queue_id, stream->stream_id);
//             stream->owner->front_end_sim_recv(
//                 0, Sys::dummy_data, msg_size, UINT8, left, stream->stream_id,
//                 &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE, &Sys::handleEvent,
//                 ehd);

//             state = State::SentRowToParent; // parent == left in our chain view
//         } else {
//             // We're row root: proceed to column reduction phase
//             row_reduced = true;
//             state = State::RowReducedAtRoot;
//         }
//         return;
//     }

//     // If the row root has completed its local row reduction, it should now
//     // initiate the column reduction by posting a recv from its down neighbor
//     // (if any) and/or starting send to up.
//     if (state == State::RowReducedAtRoot) {
//         // Post recv from down neighbor to accumulate column reductions
//         if (down >= 0) {
//             sim_request rcv_req;
//             rcv_req.vnet = this->stream->current_queue_id;
//             RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
//                 stream, stream->owner->id, EventType::PacketReceived,
//                 stream->current_queue_id, stream->stream_id);
//             stream->owner->front_end_sim_recv(
//                 0, Sys::dummy_data, msg_size, UINT8, down, stream->stream_id,
//                 &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE, &Sys::handleEvent,
//                 ehd);
//         }
//         // if we are not column root (y > 0), we will send upward after receiving
//         // from below (or immediately if no below).
//         state = State::WaitingForColumnChild;
//         return;
//     }

//     // --- Column reduction ---
//     if (state == State::WaitingForColumnChild && event == EventType::PacketReceived) {
//         // Received from below: perform reduction locally
//         (new PacketBundle(stream->owner, stream, true, false, msg_size,
//                           MemBus::Transmition::Usual))
//             ->send_to_NPU();

//         // If we have an up neighbor, forward the reduced column value up.
//         if (up >= 0) {
//             sim_request snd_req;
//             snd_req.srcRank = stream->owner->id;
//             snd_req.dstRank = up;
//             snd_req.tag = stream->stream_id;
//             snd_req.reqType = UINT8;
//             snd_req.vnet = this->stream->current_queue_id;
//             stream->owner->front_end_sim_send(
//                 0, Sys::dummy_data, msg_size, UINT8, up, stream->stream_id,
//                 &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE, &Sys::handleEvent,
//                 nullptr);

//             // register recv from up for later broadcast
//             sim_request rcv_req;
//             rcv_req.vnet = this->stream->current_queue_id;
//             RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
//                 stream, stream->owner->id, EventType::PacketReceived,
//                 stream->current_queue_id, stream->stream_id);
//             stream->owner->front_end_sim_recv(
//                 0, Sys::dummy_data, msg_size, UINT8, up, stream->stream_id,
//                 &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE, &Sys::handleEvent,
//                 ehd);

//             state = State::SentColumnToParent;
//         } else {
//             // We're the global root (0,0): column reduced result available
//             col_reduced = true;
//             // start broadcast downwards: if we have down neighbors, send
//             if (down >= 0) {
//                 sim_request snd_req;
//                 snd_req.srcRank = stream->owner->id;
//                 snd_req.dstRank = down;
//                 snd_req.tag = stream->stream_id;
//                 snd_req.reqType = UINT8;
//                 snd_req.vnet = this->stream->current_queue_id;
//                 stream->owner->front_end_sim_send(
//                     0, Sys::dummy_data, msg_size, UINT8, down, stream->stream_id,
//                     &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE, &Sys::handleEvent,
//                     nullptr);
//             }
//             state = State::BroadcastingColumnDown;
//         }
//         return;
//     }

//     // When a node receives the broadcast from up neighbor, it should forward it
//     // down to its child (if any) and then start the row-level broadcast to the
//     // right.
//     if (state == State::BroadcastingColumnDown && event == EventType::PacketReceived) {
//         // Received the global reduced result from up (or if this node is the
//         // global root it may not receive any packet but the state handles sends)
//         // Perform any local-copy action (PacketBundle -> send_to_NPU to place
//         // the final data in local memory if desired).
//         (new PacketBundle(stream->owner, stream, false, false, msg_size,
//                           MemBus::Transmition::Usual))
//             ->send_to_NPU();

//         // Forward down if needed
//         if (down >= 0) {
//             sim_request snd_req;
//             snd_req.srcRank = stream->owner->id;
//             snd_req.dstRank = down;
//             snd_req.tag = stream->stream_id;
//             snd_req.reqType = UINT8;
//             snd_req.vnet = this->stream->current_queue_id;
//             stream->owner->front_end_sim_send(
//                 0, Sys::dummy_data, msg_size, UINT8, down, stream->stream_id,
//                 &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE, &Sys::handleEvent,
//                 nullptr);
//         }

//         // After we have the final reduced value locally, start the row-wise
//         // broadcast to the right so every node in the row gets the final result.
//         if (right >= 0) {
//             sim_request snd_req;
//             snd_req.srcRank = stream->owner->id;
//             snd_req.dstRank = right;
//             snd_req.tag = stream->stream_id;
//             snd_req.reqType = UINT8;
//             snd_req.vnet = this->stream->current_queue_id;
//             stream->owner->front_end_sim_send(
//                 0, Sys::dummy_data, msg_size, UINT8, right, stream->stream_id,
//                 &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE, &Sys::handleEvent,
//                 nullptr);
//         }

//         // Once row broadcast is sent, this node is done and can exit once its
//         // own receive (if any) is processed. For simplicity we mark End when
//         // we both forwarded down (if any) and sent right (if any).
//         state = State::BroadcastingRowRight;
//         return;
//     }

//     if (state == State::BroadcastingRowRight && event == EventType::PacketReceived) {
//         // This node received the row-level broadcast from left; write to NPU
//         (new PacketBundle(stream->owner, stream, false, false, msg_size,
//                           MemBus::Transmition::Usual))
//             ->send_to_NPU();

//         // If we have a right neighbor, forward it
//         if (right >= 0) {
//             sim_request snd_req;
//             snd_req.srcRank = stream->owner->id;
//             snd_req.dstRank = right;
//             snd_req.tag = stream->stream_id;
//             snd_req.reqType = UINT8;
//             snd_req.vnet = this->stream->current_queue_id;
//             stream->owner->front_end_sim_send(
//                 0, Sys::dummy_data, msg_size, UINT8, right, stream->stream_id,
//                 &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE, &Sys::handleEvent,
//                 nullptr);
//         }

//         // Done with the collective
//         state = State::End;
//         exit();
//         return;
//     }

//     // handle simple send-complete or general events that don't carry PacketReceived
//     if (state == State::SentRowToParent && event == EventType::General) {
//         // after sending our row reduced value to parent (left), we expect no
//         // more actions until we later receive the row broadcast from left.
//         // nothing to do here for now.
//         return;
//     }

//     if (state == State::SentColumnToParent && event == EventType::General) {
//         // similarly, wait for broadcast from up
//         return;
//     }

//     // default: if we reach End, call exit() to finish stream
//     if (state == State::End) {
//         exit();
//         return;
//     }
// }
