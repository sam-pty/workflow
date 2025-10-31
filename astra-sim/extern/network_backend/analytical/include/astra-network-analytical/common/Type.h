/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <cstdint>
#include <vector>

namespace NetworkAnalytical {

/// Callback function pointer: "void func(void*)"
using Callback = void (*)(void*);

/// Callback function argument: void*
using CallbackArg = void*;

/// Device ID which starts from 0
using DeviceId = int;

/// Chunk size in Bytes
using ChunkSize = uint64_t;

/// Bandwidth in GB/s
using Bandwidth = double;

/// Latency in ns
using Latency = double;

/// Event time in ns
using EventTime = uint64_t;

/// Basic multi-dimensional topology building blocks
enum class TopologyBuildingBlock {
    Undefined,
    Ring,
    FullyConnected,
    Switch,
    Bus,
    BinaryTree,
    DoubleBinaryTree,
    Mesh,
    HyperCube,
    Torus2D,
    Mesh2D,
    KingMesh2D
};

/// Multi-dimensional address of a device.
/// Each NPU ID can be broken down into multiple dimensions.
/// for example, if the topology size is [2, 8, 4] and the NPU ID is 31,
/// then the NPU ID can be broken down into [1, 7, 1].
using MultiDimAddress = std::vector<DeviceId>;

enum class ConnectionType { Dedicated, Shared };
/// Connection policy between two devices, (src, dst) means a link from src to dst
struct ConnectionPolicy {

    DeviceId src;
    DeviceId dst;

    ConnectionPolicy(DeviceId src, DeviceId dst) : src(src), dst(dst){};
};

}  // namespace NetworkAnalytical
