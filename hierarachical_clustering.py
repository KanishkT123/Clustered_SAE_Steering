import torch
import torch.nn.functional as F
import urllib.parse
import json

def hook_fn(residual_outputs, module, input, output):
    residual_outputs.append(output)
  
def flatten_and_cosine_sim(tensor1, tensor2):
    # Flatten the tensors
    flattened1 = tensor1.view(tensor1.size(0), -1)
    flattened2 = tensor2.view(tensor2.size(0), -1)

    if flattened1.shape != flattened2.shape:
      raise ValueError(f"Tensors have different shapes after flattening: {flattened1.shape} vs {flattened2.shape}")

    # Normalize the flattened tensors
    normalized1 = F.normalize(flattened1, p=2, dim=1)
    normalized2 = F.normalize(flattened2, p=2, dim=1)

    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(normalized1, normalized2)

    return cosine_sim

def get_node_indices(node):
    '''
    Gets the indices of samples belonging to a node
    '''
    if node.is_leaf():
        return [node.id]
    else:
        left_indices = get_node_indices(node.left)
        right_indices = get_node_indices(node.right)
        return left_indices + right_indices

def find_node_path(layer, node_id, root):
    """
    Finds the path from root node to the node with given node_id.
    Returns a list of choices ('left' or 'right') to traverse the path.
    """
    def traverse(node, path=''):
        if node is None:
            return None
        if node.id == node_id:
            return path
        left_path = traverse(node.left, path + 'L')
        right_path = traverse(node.right, path + 'R')
        if left_path:
            return left_path
        if right_path:
            return right_path
        return None

    return traverse(root)

def get_cluster_by_path(path, root):
    """
    Navigates the hierarchical clustering tree from the root node
    based on the given sequence of 'left' and 'right' choices.
    Returns the cluster node reached after following the path.
    """
    node = root
    for direction in path:
        if direction == 'L':
            node = node.left
        elif direction == 'R':
            node = node.right
        else:
            raise ValueError("Invalid direction: {}".format(direction))
    return node

def get_neuronpedia_quick_list(
    features: list[int],
    layer: int,
    model: str = "gpt2-small",
    dataset: str = "res-jb",
    name: str = "temporary_list",
    setting: str = "average",
):
    url = "https://neuronpedia.org/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {"modelId": model, "layer": f"{layer}-{dataset}", "index": str(feature)}
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    print(url)
    return url

def build_cluster(layer, roots, feature_id, height=3, setting = 'average', verbose=True):
    layer = int(layer)
    root = roots[setting][layer]
    node_path = find_node_path(layer, feature_id, root)
    cluster_path = node_path[:-height]
    cluster = get_cluster_by_path(cluster_path, root=root)
    indices = get_node_indices(cluster)
    list_name = f'height {height} above L{layer}f{feature_id} with cluster setting: {setting}'
    url = get_neuronpedia_quick_list(indices, layer, name=cluster_path)
    if verbose:
        print(f'path to node: {node_path}')
        print(f'path to cluster: {cluster_path}')
        print(f'features in cluster: {indices}')
    return indices
