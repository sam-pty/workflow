/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/DoubleBinaryTree.h"

#include <cassert>
#include <cmath>
#include <iostream>

namespace NetworkAnalyticalCongestionAware
{

DoubleBinaryTree::DoubleBinaryTree(const int npus_count,
                                   const Bandwidth bandwidth,
                                   const Latency latency,
                                   const bool is_multi_dim) noexcept
    : BasicTopology(npus_count, npus_count, bandwidth, latency) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // set topology type
    this->basic_topology_type = TopologyBuildingBlock::DoubleBinaryTree;

    // initialize the root node
    // std:: cout << "npus_count: " << npus_count << std::endl;
    int depth = static_cast<int>(std::ceil(std::log2(npus_count + 1)) - 1);
    // std::cout << "Depth of the tree: " << depth << std::endl;
    m_root_max_tree_root = new Node();
    m_root_min_tree_root = new Node();
    m_node_list.push_back(m_root_max_tree_root);
    m_node_list.push_back(m_root_min_tree_root);
    m_root_max_tree_root->left = initialize_tree(depth - 1, npus_count - 1); // create the root node, leaf is at depth 0
    m_root_min_tree_root->right = initialize_tree(depth - 1, npus_count - 1); // create the root node, leaf is at depth 0

    // assign id's
    build_tree(m_root_max_tree_root, m_max_start);
    build_tree(m_root_min_tree_root, m_min_start);

    // traverse the tree and connect nodes
    if (!is_multi_dim) {
        connect_nodes(m_root_max_tree_root, bandwidth, latency);
        connect_nodes(m_root_min_tree_root, bandwidth, latency);
    }

    m_min_start = 0; // reset the starting id for the first node
    m_max_start = 0; // reset the starting id for the first node
    // print(m_root_max_tree_root);
    // print(m_root_min_tree_root);
}

DoubleBinaryTree::~DoubleBinaryTree() {
    for (auto n : m_node_list) {
        delete n;
    }
    m_node_list.clear();
}


Route DoubleBinaryTree::route(DeviceId src, DeviceId dest) const noexcept {
    // assert npus are in valid range
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    // construct empty route
    auto route = Route();

    const auto path_max_tree = get_path(m_root_max_tree_root, src, dest);
    const auto path_min_tree = get_path(m_root_min_tree_root, src, dest);

    // choose the shorter path, if equal, prefer min tree
    const auto path = path_max_tree.size() < path_min_tree.size() ? path_max_tree : path_min_tree;

    for (const auto& id : path) {
        // push the device corresponding to the id into the route
        route.push_back(devices[id]);
        // std::cout << "Adding device with ID: " << id << std::endl;
    }

    // std::cout << "Route: ";
    // for (auto r : route)
    // {
    //     std::cout << r->get_id() << " -> ";
    // }
    // std::cout << std::endl;

    // return the constructed route
    return route;
}

std::vector<ConnectionPolicy> DoubleBinaryTree::get_connection_policies() const noexcept {
    return m_policies;
}

Node* DoubleBinaryTree::initialize_tree(uint32_t depth, uint32_t total_npus_left)
{
    if (total_npus_left <= 0) {
        return nullptr; // base case: no nodes to create
    }
    Node * node = new Node();
    uint32_t max_left_use = (1 << depth) - 1; // 2 ^ n - 1, where n is the depth
    if (total_npus_left > max_left_use) {
        // std::cout << "left branch " << max_left_use << std::endl;
        // std::cout << "right branch: " << total_npus_left - 1 - max_left_use << std::endl;
        node->left  = initialize_tree(depth - 1, max_left_use);
        node->right = initialize_tree(depth - 1, total_npus_left - 1 - max_left_use);
    }
    else
    {
        node->left = initialize_tree(depth - 1, total_npus_left - 1);
    }
    
    return node;
}

void DoubleBinaryTree::build_tree(Node* node, uint32_t& curr_id) {
    // inorder traversal
    // left -> node -> right
    if (node->left != nullptr) {
        build_tree(node->left, curr_id);
    }
    node->id = curr_id;
    // std::cout << "Assigning ID: " << m_start << " to node." << std::endl;
    curr_id++;
    m_node_list.push_back(node);
    if (node->right != nullptr) {
        build_tree(node->right, curr_id);
    }
}

void DoubleBinaryTree::connect_nodes(Node* node, Bandwidth bandwidth, Latency latency)
{
    if (node->left != nullptr) {
        // could be bidirectional or two wires
        // now two wires
        try
        {
            connect(node->left->id, node->id, bandwidth, latency, true);
            // add connection policy
            m_policies.emplace_back(node->left->id, node->id);
            m_policies.emplace_back(node->id, node->left->id);
        }
        catch (const std::runtime_error& e)
        {
            std::cerr << e.what() << '\n';
        }
        
        connect_nodes(node->left, bandwidth, latency);
    }
    if (node->right != nullptr) {
        connect(node->right->id, node->id, bandwidth, latency, true);
        // add connection policy
        m_policies.emplace_back(node->right->id, node->id);
        m_policies.emplace_back(node->id, node->right->id);
        // recurse to subtree
        connect_nodes(node->right, bandwidth, latency);
    }
}

bool DoubleBinaryTree::find_path(Node* node, int target_id, std::vector<int>& path) const
{
    if (!node) return false;

    path.push_back(node->id);

    if (node->id == target_id) return true;

    if (find_path(node->left, target_id, path) || find_path(node->right, target_id, path)) {
        return true;
    }

    path.pop_back();
    return false;
}

std::vector<int> DoubleBinaryTree::get_path(Node* root, int source_id, int dest_id) const{
    std::vector<int> path_to_source, path_to_dest;
    
    // Find paths from root to each node
    find_path(root, source_id, path_to_source);
    find_path(root, dest_id, path_to_dest);

    // std:: cout << "----------------------------" << std::endl;
    // std::cout << "root id is: " << root->id << std::endl;
    // std::cout << "source id is: " << source_id << std::endl;
    // std::cout << "Path to source:";
    // for (const auto& id : path_to_source) {
    //     std::cout << " " << id;
    // }
    // std::cout << std::endl;
    // std::cout << "root id is: " << root->id << std::endl;
    // std::cout << "dest id is: " << dest_id << std::endl;
    // std::cout << "Path to destination: ";
    // for (const auto& id : path_to_dest) {
    //     std::cout << " " << id;
    // }
    // std::cout << std::endl;

    // Find Lowest Common Ancestor (LCA)
    int lca_index = 0;
    while (lca_index < path_to_source.size() &&
           lca_index < path_to_dest.size() &&
           path_to_source[lca_index] == path_to_dest[lca_index]) {
        lca_index++;
    }

    std::vector<int> path;

    // Add path from source up to (but not including) LCA
    for (int i = path_to_source.size() - 1; i >= lca_index; --i) {
        path.push_back(path_to_source[i]);
    }

    // Add path from LCA down to destination (including LCA)
    for (int i = lca_index - 1; i < path_to_dest.size(); ++i) {
        path.push_back(path_to_dest[i]);
    }

    // std::cout << "Final path: ";
    // for (const auto& id : path) {
    //     std::cout << id << " ";
    // }
    // std::cout << std::endl;
    
    // std::cout << "----------------------------" << std::endl;

    return path;
}

void DoubleBinaryTree::print(Node* node) const
{
    if (node == nullptr) return;
    // in order traversal
    print(node->left);
    std::cout << "print Tree Node ID: " << node->id << std::endl;
    print(node->right);
}

};
// namespace NetworkAnalyticalCongestionAware
