import transformers
import torch
# from modelscope import snapshot_download
import json
import re
import sys
import pickle
import os
from tqdm import tqdm
import random

import datetime
import time
import numpy as np

def new_log(log_path, args):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    filename = os.path.join(log_path, timestamp)
    for k, v in args.items():
        arg_text = f'-{k}[{v}]'
        filename += arg_text
    f = open(filename + ".txt", "w")
    print("filename == ", filename)
    return f
    

# llama8b = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
# mistral7b8 = "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1"
# model_id = snapshot_download(llama8b)

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device=torch.device("cuda:6")
# )
from openai import OpenAI
def llm1(system_prompt,prompt):
    client = OpenAI(
    base_url = 'http://localhost:11433/v1',
    api_key='ollama', # required, but unused
    )
    response = client.chat.completions.create(
        model="llama3.1:70b",
        messages=[
        {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
  ]
)
    #print(response.usage)
    return response.choices[0].message.content



tot_task = ["common neighbor", "connected components computation", "shortest path between two nodes", "graph diameter", "maximum independent set", "minimum vertex cover", "maximum clique", "maximum common induced subgraph", "travelling salesman"]
tot_dataset_name = ["Neighbor", "Connected", "Distance", "Diameter", "MIS", "MVC", "MCP", "MCS", "TSP"]
tot_diff = ["easy", "hard"]

args = {
    "task" : "connected components computation",
    "dataset_name" : "Connected",
    "diff" : "easy"
}


NP = ["MIS", "MVC", "TSP", "MCS", "MCP"]
pred_less_groundtruth = ["MIS", "MCS", "MCP"]
pred_greater_groundtruth = ["MVC", "TSP"]

def work(args):
    if not os.path.exists("./log_llama70b"):
        os.mkdir("./log_llama70b")
    logger = new_log("./log_llama70b", args)
    
    def PRINT_LOG(text):
        print(text)
        print(text, file=logger)
        
    task = args["task"]
    dataset_name = args["dataset_name"]
    diff = args["diff"]
    system_prompt = f"You are an expert in the field of graph algorithms and are currently solving {task} problem. I will provide a series of problems, and please try to solve these problems step by step and give the final answers in the following format: 'The answer is [number]'. Please make ensure that the output in the last line is 'The answer is [number]'. [number] is filled with your answer."

    data = pickle.load(open(f"./dataset/{dataset_name}_{diff}.pkl", "rb"))

    tot_number = 0
    feasible = 0
    correct = 0
    ratio = []
    random.shuffle(data)
    t1 = time.time()
    for x in tqdm(data):
        user_content = x["text"]
        exact_answer = x["exact_answer"]
        if exact_answer is not None:
            exact_answer = int(exact_answer)
            # messages = [
            #     {"role": "system", "content": system_prompt},
            #     {"role": "user", "content": user_content}
            # ]
            # outputs = pipeline(
            #     messages,
            #     max_new_tokens=2048,
            # )
            content=llm1(system_prompt,user_content)
            # content = outputs[0]["generated_text"][-1]["content"]
            result = content.strip('.').split(' ')[-1]
            
            tot_number += 1
            try:
                result = int(result)
            except Exception as e:
                PRINT_LOG(f"result == {result}")
                PRINT_LOG(f"Exception: {e}")
            else:
                # tot_number += 1
                PRINT_LOG(f"result == {result}, exact_answer == {exact_answer}")
                if (result == exact_answer):
                    correct += 1
                if dataset_name in pred_less_groundtruth:
                    feasible += (result <= exact_answer)
                    if exact_answer > 0:
                        ratio.append(abs(result - exact_answer) / abs(exact_answer))
                elif dataset_name in pred_greater_groundtruth:
                    feasible += (result >= exact_answer)
                    if exact_answer > 0:
                        ratio.append(abs(result - exact_answer) / abs(exact_answer))
    t2 = time.time()
    PRINT_LOG(f"average time: {(t2 - t1) / tot_number:.2f}s")
    PRINT_LOG(f"correct: {correct / tot_number * 100:.2f}%")
    if dataset_name in NP:
        PRINT_LOG(f"feasible: {feasible / tot_number * 100:.2f}%")
        PRINT_LOG(f"ratio: {np.mean(ratio):.4f}")
        
        
if __name__ == "__main__":
    for i in range(0, len(tot_task)):
        task = tot_task[i]
        dataset_name = tot_dataset_name[i]
        for diff in tot_diff:
            args = {
                "task": task,
                "dataset_name": dataset_name,
                "diff": diff
            }
            work(args)
