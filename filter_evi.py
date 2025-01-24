import random
import pickle
from constants import PATH_DB,PATH_DATA,account_id,headers,API_BASE_URL,cloudfare_api
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import requests
from multiprocessing import Pool
from functools import partial
import csv
import re
import json
import random

import os
class KG():
    def __init__(self, kg):
        super().__init__()
        self.kg = kg
        
    def get_one_hop_subgraph(self, entity):
        """
        Extract the one-hop subgraph for the given entity.
    
        Args:
            entity (str): The entity for which to extract the subgraph.
    
        Returns:
            dict: A dictionary with keys as relation, containing the respective neighbors.
        """
        subgraph = {}
        if entity in self.kg:
            for rel,tails in self.kg[entity].items():
                subgraph[rel]=tails
            
        return subgraph
def load_db(path):
    """
    Load the knowledge base
    """
    with open(path, 'rb') as file:
        dbpedia = pickle.load(file)
    return dbpedia

def get_samples(num,split,file_num):
    if split=="train":
        sub_path="train"
    elif split=="val":
        sub_path="dev"
    elif split=="test":
        sub_path="test"
    else:
        raise ValueError("Invalid split name")
    with open(path, 'rb') as file:
        data_raw = pickle.load(file)
    random.seed(42)  # Set a seed for reproducibility
    selected_keys = random.sample(list(data_raw.keys()), num)
    selected_data = {key: data_raw[key] for key in selected_keys}

    part_size = len(selected_data) // file_num  # Size of each part (2,000 in this case)
    parts = []
    keys = list(selected_data.keys())
    start = 0
    for i in range(file_num):
        end = start + part_size
    # Handle any leftover keys in the last part
        if i == file_num-1:
           end = len(selected_data)
        part = {key: selected_data[key] for key in keys[start:end]}
        parts.append(part)
        start = end

# Save each part into separate pickle files
    for i, part in enumerate(parts):
        with open(f'./data/{split}_data_part_{i+1}.pickle', 'wb') as file:
            pickle.dump(part, file)

def get_one_hop(kg,split,file_num):

    for i in range(file_num):
        with open(f'./data/{split}_data_part_{i+1}.pickle', 'rb') as file:
            test_pedia = pickle.load(file)
        evidence={}
        for claim, information in test_pedia.items():
            tags=information["types"]
            resoning_types=[]
            for tag in tags:
                if tag == 'negation':
                    resoning_types.append('negation')
                    break
                elif tag == 'num1':
                    resoning_types.append('one-hop')
                elif tag == 'multi claim':
                    resoning_types.append('conjuction')
                elif tag == 'existence':
                    resoning_types.append('existence')
                elif tag == 'multi hop':
                    resoning_types.append('multi hop')
            subevidence = {
        "entity_set":{},
        "evidence_onehop_full":{},
        "label":information["Label"],
        "reasoning_types":resoning_types
            }
            subevidence["entity_set"]=information['Entity_set']
            for entity in information['Entity_set']:
                subevidence["evidence_onehop_full"][entity]=kg.get_one_hop_subgraph(entity)
            evidence[claim]=subevidence

        output_path= f'./data/{split}_evi_part_{i+1}.pickle'   
        with open(output_path, 'wb') as output_file:
            pickle.dump(evidence, output_file)



def process_response(text,information):
    processed_text = text.strip().strip("```python").strip("```").strip()
    processed_text=processed_text.replace("```", "")
    output_dict={}
    valid_output={}
# Convert to a dictionary
    try:
        output_dict = eval(processed_text)
    except Exception as e:
        valid_output={"invalid output"}
    try:
        for entity, relations in output_dict.items():
            if entity in information['entity_set']:
                for relation in relations:
                    if relation in information['evidence_onehop_full'][entity]:
                        valid_output[entity]=relations
    except Exception as e:
        valid_output={"invalid output"}
    return valid_output


def call_llm(claim, entities,evidence):
    entities_filtered = {entity.replace('"', '') for index, entity in enumerate(entities,start=1)}
    output_expectations= "{\n\n" + "".join([f'''"{entity}-{index}": ["..." , "...", ... ],  # options (strictly choose no more than 5 from): ''' + " , ".join(random.sample(list(connections), min(len(connections), 15))) + "\n\n" for index,(entity, connections) in evidence.items()]) + "}"
    
    content = f'''
    Claim:
    {claim}
    '''
    message= [{"role": "system", "content": 
    '''
    You are an intelligent graph relation finder. You are given a single claim and all connections of the entities in the claims, your task is to filter out the connections that are related to the claim that helps fact-checking. "~" beginning connection means reverse connection.'''
 },{"role": "user", "content": content+ '''
    ## TASK:
     - For each of the given entities given below: 
       Filter the connections strictly from the given options that would be relevant to connect given entities to fact-check Claim1.
    - Think clever, there could be multi-step hidden connections, if not direct, that could connect the entities somehow.
    - Arrange them based on their relevance. Be extra careful with ~ signs.
    - No code output. No explanation. Output only valid python DICT of structure:\n'''+ output_expectations}]

    return message

def run(model, inputs):
    input = { "messages": inputs }
    response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
    return response.json()

def send_request(inputs):
    output = run("@cf/meta/llama-3.2-3b-instruct", inputs)
    response = output['result']['response']
   # print(response)
    return response

def write_to_csv(evidence_filtered, filename):
    """
    Write the evidence_filtered dictionary to a CSV file.
    
    Args:
        evidence_filtered (dict): The dictionary containing claim information.
        filename (str): The path to the CSV file.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['Claim', 'Evidence_Filtered'])
        
        # Write the content
        for claim, information in evidence_filtered.items():
            writer.writerow([
                claim,
                '; '.join(information.get('entity_set', [])),  # Serialize entity_set as a semicolon-separated string
                json.dumps(information.get('evidence_filtered', {}), ensure_ascii=False)  # Serialize evidence_filtered as JSON
            ])

# Initialize filename for the CSV
csv_filename = "E://evidence_filtered_train_1.csv"

i=0
data=test_set
exceed=0
evidence_filtered={}
for claim, information in data.items():
    i+=1
    print(f"Claim {i} Processing")
    count=0
    valid_output = False
    while (not valid_output):
        message = call_llm(claim, information['entity_set'], information['evidence_onehop_full'])
        text = send_request(message)
        output = process_response(text, information)
        if output != {'invalid output'}:
            valid_output = True  # Exit the loop if the output is valid
            information['evidence_filtered']=output
        else:
            if count <5:
                print(f"Invalid output for claim {claim}, retrying...")
                count+=1
            else:
                print(f"Exceed trying time")
                exceed+=1
                break
    evidence_filtered[claim]=information
    # Call the function to write to CSV after processing each claim
    write_to_csv(evidence_filtered, csv_filename)
output_path_2= 'E://traintset_filtered_1.pickle'
with open(output_path_2, 'wb') as output_file:
        pickle.dump(evidence_filtered, output_file)

print(exceed)#305 out of 9041

def get_filter_result(dbpedia,split):
    if split=="train":
        sub_path="train"
        num=10
    elif split=="val":
        num=2
        sub_path="dev"
    elif split=="test":
        sub_path="test"
        num=2
    else:
        raise ValueError("Invalid split name")
    for i in range(num):
        with open(f'./data/{split}set_filtered_{i+1}.pickle', 'rb') as file:
            data_raw = pickle.load(file)
# Process all entities and save two-hop neighbors
        adddirect={}
        for claim, information in data_raw.items():
            information['direct']={}
            information['filtered']={}
            direct_connection_found = False
            for entity in information['entity_set']:
                if entity in dbpedia:
                    information['direct'][entity]={}
                    for relation, neighbours in dbpedia[entity].items():
                        for neighbour in neighbours:
                            if neighbour in information['entity_set']:
                                information['direct'][entity][relation]=neighbours
                                direct_connection_found = True
                else:
                    print(f"{entity} could not be found")
            if 'evidence_filtered' in information:
                for entity_,relations in information['evidence_filtered'].items():
                    information['filtered'][entity_]={}
                    relations=information['evidence_filtered'][entity_]
                    for relation in relations: 
                        if  relation in information['evidence_onehop_full'][entity_]:
                            information['filtered'][entity_][relation]=random.sample(
                            information['evidence_onehop_full'][entity_][relation],  # Source list
                        min(len(information['evidence_onehop_full'][entity_][relation]), 3)  # Max 3 items
                    )
            else:
                information['evidence_filtered']={}
            del information['evidence_onehop_full']
            if not direct_connection_found:
                information['direct'] = None
        
            adddirect[claim]=information
        output_path=f'./data/{split}set_final_{i+1}.pickle'
        with open(output_path_2, 'wb') as output_file:
            pickle.dump(adddirect, output_file)

    