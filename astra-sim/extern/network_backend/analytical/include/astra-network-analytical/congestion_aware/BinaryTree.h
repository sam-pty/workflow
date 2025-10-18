/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/BasicTopology.h"

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

struct Node {
  Node* left = nullptr;
  Node* right = nullptr;
  int id = -1; // id is based on inorder traversal
};

/**
 * Implements a binary tree topology.
 *
 * Example:
 * 
 *           7
 *          /
 *         3
 *       /   \
 *      1     5
 *     / \   / \
 *    0   2 4   6
 *
 * Therefore, the number of NPUs and devices are both 8.
 *
 */
class BinaryTree final : public BasicTopology {
public:
  /**
   * Constructor.
   *
   * @param npus_count number of npus in a ring
   * @param bandwidth bandwidth of link
   * @param latency latency of link
   * @param bidirectional true if ring is bidirectional, false otherwise
   */
  BinaryTree(int npus_count, Bandwidth bandwidth, Latency latency, const bool is_multi_dim = false) noexcept;

  ~BinaryTree() override;

  /**
   * Implementation of route function in Topology.
   */
  [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;

  /**
   * Get connection policies
   *
   * @return list of connection policies
   */
  [[nodiscard]] std::vector<ConnectionPolicy> get_connection_policies() const noexcept override;

private:
  /** Builds the binary tree recursively.
   * The root node is at depth 0, and the left child is at depth 1.
   */
  Node* initialize_tree(uint32_t depth, uint32_t total_npus_left);

  /** Builds the binary tree and assigns ids to nodes based on inorder traversal.
   * The ids are assigned starting from 0.
   */
  void build_tree(Node* node);

  /** Connects nodes in the binary tree.
   * 
   * @param node the current node to connect
   * @param bandwidth the bandwidth of the link
   * @param latency the latency of the link
   */
  void connect_nodes(Node* node, Bandwidth bandwidth, Latency latency);


  /** Finds a path from the root node to the target node.
   * 
   * @param node the current node to check
   * @param target_id the id of the target node
   * @param path a vector to store the path
   * @return true if the path is found, false otherwise
   */
  bool find_path(Node* node, int target_id, std::vector<int>& path) const;

  /** Gets the path from source to destination in the binary tree.
   * 
   * @param root the root node of the tree
   * @param source_id the id of the source node
   * @param dest_id the id of the destination node
   * @return a vector containing the ids of the nodes in the path
   */
  std::vector<int> get_path(Node* root, int source_id, int dest_id) const;

  /** Prints the binary tree in inorder traversal.
   * 
   * @param node the current node to print
   */
  void print(Node* node) const;

  /// List of nodes in the binary tree.
  std::vector<Node*> m_node_list;
  /// The root node of the binary tree.
  Node* m_root = nullptr;
  /// indexing used to assign ids to nodes
  uint32_t m_start = 0; // starting id for the first node
  /// connection policies constructed while building tree
  std::vector<ConnectionPolicy> m_policies;
};

}  // namespace NetworkAnalyticalCongestionAware
