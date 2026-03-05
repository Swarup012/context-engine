"""Traverse the dependency graph to find hot/warm/cold candidates."""

from typing import Set

import networkx as nx


def get_context_candidates(
    focal_qualified_name: str,
    graph: nx.DiGraph,
    depth: int = 2
) -> tuple[list[str], list[str], list[str]]:
    """
    Get hot/warm/cold candidates from the dependency graph.
    
    Args:
        focal_qualified_name: The focal function's qualified name.
        graph: NetworkX DiGraph of function dependencies.
        depth: Maximum depth for neighbor search (default 2).
        
    Returns:
        Tuple of three lists:
        - hot_candidates: focal node + 1-hop neighbors
        - warm_candidates: 2-hop neighbors
        - cold_candidates: everything else in the graph
    """
    # Check if focal node exists in graph
    if focal_qualified_name not in graph:
        # Return empty hot/warm, all nodes as cold
        all_nodes = list(graph.nodes())
        return [], [], all_nodes
    
    # Hot: focal node + 1-hop neighbors
    hot_set: Set[str] = {focal_qualified_name}
    
    # Get 1-hop neighbors (functions it calls + functions that call it)
    # Predecessors: functions that call the focal function
    predecessors = set(graph.predecessors(focal_qualified_name))
    # Successors: functions that the focal function calls
    successors = set(graph.successors(focal_qualified_name))
    
    one_hop = predecessors | successors
    hot_set.update(one_hop)
    
    # Warm: 2-hop neighbors (neighbors of neighbors)
    warm_set: Set[str] = set()
    
    if depth >= 2:
        for neighbor in one_hop:
            # Get neighbors of this neighbor
            neighbor_predecessors = set(graph.predecessors(neighbor))
            neighbor_successors = set(graph.successors(neighbor))
            two_hop = neighbor_predecessors | neighbor_successors
            
            # Add to warm if not already in hot
            for node in two_hop:
                if node not in hot_set:
                    warm_set.add(node)
    
    # Cold: everything else in the graph
    all_nodes = set(graph.nodes())
    cold_set = all_nodes - hot_set - warm_set
    
    return list(hot_set), list(warm_set), list(cold_set)
def traverse_multi_focal(
    focal_names: list[str],
    graph: nx.DiGraph,
    depth: int = 2
) -> tuple[list[str], list[str], list[str]]:
    """
    Traverse from multiple focal points and merge results intelligently.
    
    Args:
        focal_names: List of focal point qualified names.
        graph: NetworkX DiGraph of function dependencies.
        depth: Maximum depth for neighbor search.
        
    Returns:
        Tuple of (hot_candidates, warm_candidates, cold_candidates).
        
    Priority rules:
    - If a function is HOT in any traversal -> HOT in merged result
    - If a function is WARM in any traversal -> WARM in merged result
    - Everything else -> COLD
    """
    all_hot = set()
    all_warm = set()
    all_cold = set()
    
    # Run traversal from each focal point
    for focal_name in focal_names:
        hot, warm, cold = get_context_candidates(focal_name, graph, depth)
        all_hot.update(hot)
        all_warm.update(warm)
        all_cold.update(cold)
    
    # Merge with priority: HOT > WARM > COLD
    # Remove from WARM any that are in HOT
    final_warm = all_warm - all_hot
    
    # Remove from COLD any that are in HOT or WARM
    final_cold = all_cold - all_hot - final_warm
    
    return list(all_hot), list(final_warm), list(final_cold)

