import transformers
import torch
from modelscope import snapshot_download
import json
import re
import sys
import pickle
import os
from func_timeout import func_set_timeout
import time


def change2edgelist(g):
    node_number = g.number_of_nodes()
    G = [[] for _ in range(node_number)]
    for edge in g.edges:
        G[edge[0]].append(edge[1])
        G[edge[1]].append(edge[0])
    return G

def change2edgelist_weight(g):
    node_number = g.number_of_nodes()
    G = [[0 for _ in range(node_number)] for _ in range(node_number)]
    for edge in g.edges(data=True):
        G[edge[0]][edge[1]] = edge[2]['weight']
        G[edge[1]][edge[0]] = edge[2]['weight']
    return G
def run_shortest_path_unweighted(G, source, target, exact_answer, func):
    output = func(G, source, target)
    if (output != exact_answer):
        return 1, output
    else:
        return 0, output
    
def run_common_neighbor(G, u, v, exact_answer, func):
    output = func(G, u, v)
    if (len(output) != exact_answer):
        return 1, len(output)
    else:
        return 0, len(output)

def run_graph_edit_distance(g1, g2, exact_answer, func):
    output = func(g1, g2)
    if (output != exact_answer):
        return 1, output
    else:
        return 0, output

@func_set_timeout(100)  
def run_maximum_common_subgraph(g1, g2, exact_answer, func):
    output = func(g1, g2)
    if (output != exact_answer):
        return 1, output
    else:
        return 0, output
    
def run(G, exact_answer, func):
    output = func(G)
    if (output != exact_answer):
        return 1, output
    else:
        return 0, output
    
def read_pseudocode(task):
    path = os.path.join("./pse_pure", task+".txt")
    out = []
    with open(path, "r") as f:
        for line in f:
            out.append(line)
    return '\n'.join(out)

def read_pseudocode_nl(task):
    path = os.path.join("./pse_nl", task+".txt")
    out = []
    with open(path, "r") as f:
        for line in f:
            out.append(line)
    return ''.join(out)

def read_core_idea_nl(task):
    path = os.path.join("./pse_nl", task+".txt")
    out = []
    with open(path, "r") as f:
        ll = f.readlines()
        out.append(ll[0])
        out.append(ll[-1])
    return ''.join(out)

def remove_useless(text):
    temp = text.split('\n')
    out = []
    for line in temp:
        if not line.startswith('```'):
            out.append(line)
    return '\n'.join(out)

llama8b = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
model_id = snapshot_download(llama8b)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=torch.device("cuda:0")
)

# messages = [
#     {"role": "system", "content": "You are an advanced AI specialized in solving graph problems. Please provide a Python function that solves the problem using only pure Python or standard library modules. Do not use any external packages such as networkx. Your solution should include complete code with proper syntax and comments where necessary."},
#     {"role": "user", "content": "You are required to create a Python function that identifies all common neighbors for two specified authors in an undirected academic network represented by an adjacency list. In this network, nodes represent authors and edges represent academic collaboration relationships between authors. The input to your function will be a dictionary where keys are the names of authors and values are lists of collaborating authors, and author1,author2 for whom to find common collaborators. \
#         Your function should return a list of strings, each representing a common collaborator of the two specified authors. Present your solution in the following format, do not write example usage of the function: def find_common_neighbors(adjacency_list, author1, author2):"}
# ]

# messages = [
#     {"role": "system", "content": "As an expert in the field of graph algorithms, you will take on the role of interpreting descriptions of graph algorithms provided by the user. Based on these descriptions, you will abstract the problem into a graph algorithm issue and provide a Python code snippet that can be executed correctly. Please ensure that the Python code is executable without any comments."},
#     {"role": "user", "content": "For an undirected graph, examine the common neighbors of two nodes. The Python code will take two lists as parameters, representing the lists of neighbors for Node 1 and Node 2, respectively. The elements in the lists are integers. Please complete the following Python code so that the function returns a list containing the common neighbors of the two nodes.\
#      def common_neighbors(list1, list2):"}
# ]

with open("prompt.json", "r") as f:
    prompt = json.load(f)

tot_tasks = prompt.keys() - {"system"}
# for task in ["common_neighbors"]:   


tot_task_map = {
    "common_neighbors": "Neighbor",
    "connected_component_undirected": "Connected",
    "graph_diameter": "Diameter",
    "shortest_path_unweighted": "Distance",
    "maximum_independent_set": "MIS",
    "minimum_vertex_cover": "MVC",
    "maximum_clique": "MCP",
    "travelling_salesman": "TSP",
    "graph_edit_distance": "GED",
    "maximum_common_subgraph": "MCS"
}


task = "common_neighbors"

# messages = [
#     {"role": "system", "content": prompt["system"]},
#     {"role": "user", "content": prompt[task]}
# ]

merge_prompt = ' '.join([prompt[task][0], prompt[task][1], read_pseudocode(task), prompt[task][-1]])
# merge_prompt = ' '.join([prompt[task][0], prompt[task][1], prompt[task][-1]])
# merge_prompt = ' '.join([prompt[task][0], prompt[task][1], read_pseudocode_nl(task), prompt[task][-1]])
# merge_prompt = ' '.join([prompt[task][0], prompt[task][1], read_core_idea_nl(task), prompt[task][-1]])

print(merge_prompt)

messages = [
    {"role": "system", "content": prompt["system"]},
    {"role": "user", "content": merge_prompt}
]

tot_trial = 10
original_tot_trial = tot_trial
if not os.path.exists(f"./output_pse_code"):
    os.mkdir(f"./output_pse_code")
if not os.path.exists(f"./output_pse_code/llama3-8b"):
    os.mkdir(f"./output_pse_code/llama3-8b")
min_error = float("inf")
ans = ""
t1 = time.time()
while (tot_trial > 0): 
    tot_trial -= 1
    error = 0
    run_exception = False
    outputs = pipeline(
        messages,
        max_new_tokens=2048,
    )
    content = outputs[0]["generated_text"][-1]["content"]
    code = content
    code = remove_useless(code)
    print(code)
    try:
        exec(code)
    except Exception as e:
        print("Exception: ", e)
        continue
    
    unittest_data = pickle.load(open(f"unittest/{tot_task_map[task]}.pkl", "rb"))
    data = []
    all_pass = True
    for i, x in enumerate(unittest_data):
        if task == "graph_edit_distance" or task == "maximum_common_subgraph":
            g1, g2 = x["graph"]
        else:
            g = x["graph"]
        if task == "common_neighbors":
            u = x["node1"]
            v = x["node2"]
        if task == "shortest_path_unweighted":
            target = x["target"]
            source = x["source"]
        exact_answer = x["exact_answer"]
        if task == "travelling_salesman":
            G = change2edgelist_weight(g)
        elif task != "graph_edit_distance" and task != "maximum_common_subgraph":
            G = change2edgelist(g)
        try:
            if task == "common_neighbors":
                flag, output = run_common_neighbor(G, u, v, exact_answer, sys.modules['__main__'].__dict__[task])
            elif task == "shortest_path_unweighted":
                flag, output = run_shortest_path_unweighted(G, source, target, exact_answer, sys.modules['__main__'].__dict__[task])
            elif task == "graph_edit_distance":
                flag, output = run_graph_edit_distance(g1, g2, exact_answer, sys.modules['__main__'].__dict__[task])
            elif task == "maximum_common_subgraph":
                try:
                    flag, output = run_maximum_common_subgraph(g1, g2, exact_answer, sys.modules['__main__'].__dict__[task])
                except Exception as e:
                    print(f"Exception: {e}")
                    all_pass = False
                    run_exception = True
                    break
            else:
                flag, output = run(G, exact_answer, sys.modules['__main__'].__dict__[task])
        except Exception as e:
            print(f"Exception: {e}")
            all_pass = False
            run_exception = True
            break
        else:
            error += abs(output - exact_answer) / exact_answer
            if (flag == 1):
                all_pass = False
                print("exact_answer == ", exact_answer)
                if task == "graph_edit_distance" or task == "maximum_common_subgraph":
                    print("g1 == ", g1)
                    print("g2 == ", g2)
                else:
                    print("G == ", G)
                print("output == ", output)
                if task == "common_neighbors":
                    print("u == ", u)
                    print("v == ", v)
                if task == "shortest_path_unweighted":
                    print("source == ", source)
                    print("target == ", target)
            else:
                print(f"{i}th data pass!")

    if all_pass:
        print("unit test is all passed!")
        output_file = open(f"./output_pse_code/llama3-8b/{task}.txt", "w")
        print(code, file=output_file)
        break
    elif not run_exception:
        if (error < min_error):
            min_error = error
            ans = code


t2 = time.time()
print(f"total time: {t2 - t1}")
print(f"trial number: {original_tot_trial - tot_trial}")
if (tot_trial == 0):
    print("cannot pass all unit test!")
    print("We choose min error code")
    output_file = open(f"./output_pse_code/llama3-8b/{task}.txt", "w")
    print(ans, file=output_file)
    
    
    
