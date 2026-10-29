from collections import deque

def bfs_shortest_path(graph, start):

    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if distances[neighbor] == float("inf"):
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances

def all_pairs_shortest_paths(graph):

    return {node: bfs_shortest_path(graph, node) for node in graph}

def sm2graph(tokens):

    if not tokens:
        return None
    graph = {index: [] for index in range(len(tokens))}
    ring_labels = []
    ring_positions = []
    branch_parents = deque()
    last_parent = None

    for index, token in enumerate(tokens):
        if token.isdigit() or token.startswith("%"):
            label = int(token if token.isdigit() else token[1:])
            if label not in ring_labels:
                ring_labels.append(label)
                ring_positions.append(index)
            else:
                position = ring_labels.index(label)
                other = ring_positions[position]
                graph[other].append(index)
                graph[index].append(other)
                ring_positions[position] = None
                ring_labels[position] = None

        if index > 0:
            previous = tokens[index - 1]
            if previous != ")":
                graph[index - 1].append(index)
                graph[index].append(index - 1)
            elif branch_parents:
                parent = branch_parents.popleft()
                if parent is not None and parent >= 0:
                    graph[parent].append(index)
                    graph[index].append(parent)

        if token == "(":
            if index > 0 and tokens[index - 1] == ")":
                branch_parents.appendleft(last_parent)
            else:
                parent = index - 1
                branch_parents.appendleft(parent)
                last_parent = parent
    return graph

def shortest_path_matrix(graph, src=None, unreachable_value=10):

    if graph is None:
        return [0] if src is not None else [[0]]
    node_count = len(graph)
    if src is not None:
        if src < 0 or src >= node_count:
            return None
        distances = bfs_shortest_path(graph, src)
        return [
            distances[node] if distances[node] != float("inf") else unreachable_value
            for node in range(node_count)
        ]
    return [
        shortest_path_matrix(graph, src=index, unreachable_value=unreachable_value)
        for index in range(node_count)
    ]

if __name__ == "__main__":
    from Tokenizer import BertTokenizer

    tokenized = BertTokenizer().tokenize("CCO")
    print(shortest_path_matrix(sm2graph(tokenized)))
