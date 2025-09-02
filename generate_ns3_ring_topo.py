#!/usr/bin/env python3

def generate_ring_topology(num_nodes, bandwidth="25Gbps", latency="0.005ms", extra=0):
    """
    Generates a ring topology config string for a given number of nodes,
    including a blank line between the header and the links.

    Parameters:
    - num_nodes (int): Number of nodes in the ring.
    - bandwidth (str): Bandwidth for each link (default: "25Gbps").
    - latency (str): Latency for each link (default: "0.005ms").
    - extra (int): Extra value to append at the end of each line (default: 0).

    Returns:
    - str: The generated ring topology configuration.
    """
    config_lines = []

    # First line: num_nodes 0 num_nodes
    num_links = num_nodes # In a ring, the number of links equals the number of nodes
    config_lines.append(f"{num_nodes} 0 {num_links}\n")
    config_lines.append("\n")  # Blank line

    # Generate ring links
    for i in range(num_nodes):
        src = i
        dst = (i + 1) % num_nodes
        config_lines.append(f"{src} {dst} {bandwidth} {latency} {extra}\n")

    return ''.join(config_lines)


# Example usage
if __name__ == "__main__":
    num_nodes = 4  # Change this as needed
    config = generate_ring_topology(num_nodes)

    with open("astra-sim/extern/network_backend/ns-3/scratch/topology/ring_topology_config.txt", "w") as f:
        f.write(config)

    print("Ring topology config written to astra-sim/extern/network_backend/ns-3/scratch/topology/ring_topology_config.txt")
