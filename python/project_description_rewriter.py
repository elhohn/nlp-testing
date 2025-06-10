#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 08:56:51 2025

@author: elliot
"""

import ollama
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import python.project_contexts as context

MODEL_NAME = 'gemma3:4b'

#############################
'''
SET UP THE MODEL AND FUNCTIONS

Build the prompt
Set up multiprocessing
'''
#############################


# Prompt template
def format_prompt(description: str, program: str, context: str) -> str:
    return f"""
    You are a helpful assistant. Re-write the following PROJECT DESCRIPTION to make it a single paragraph. 
    While the PROJECT DESCRIPTION is the main source of information you should use for this task, you can also
    use the information in the PROJECT_CONTEXT section below to inform your response. Lastly, follow the INSTRUCTIONS 
    listed below very carefully. Deviation from these instructions will be strongly penalized.

    PROJECT DESCRIPTION:
    \"\"\"
    {description}
    \"\"\"

    PROJECT CONTEXT:
    This project is part of the {program} program. Here is additional information about the project: {context}.
    
    INSTRUCTIONS:
    - Return a single paragraph with complete sentences, totaling 370 words or fewer.
    - Start the first sentence of each summary with the name of the organization listed in the PROJECT DESCRIPTION.
    - Make the summary sound professional, clear, and with correct grammar, spelling, and punctuation
    - Include the following information in the summary, in this order:
        1. Purpose of the project
        2. Items being purchased or built (if applicable)
        3. Anticipated outcomes
        4. Beneficiaries (the people or communities helped by the project)
    - Describe the beneficiary populations in general terms using words such as "farmers", "consumers", "ranchers", 
        "processors", and/or "stakeholders"
    - Exclude the following words: "diversity", "underserved", "under-served", "minority", "urban", "inner city",
        "woman owned", "climate smart", "conservation", "climate change", and "biodiversity", and anything else that
        might be considered (rightly or not) to be DEI-related.
    - Exclude any discussion of training.
    - Do not include anything about the "PROJECT CONTEXT" (i.e., don't mention the program that this project is part of); focus
        only on the project in the PROJECT DESCRIPTION.
    - Use complete sentences.
    - Do not include anything about your instructions (i.e. omit anything like "as a helpful assistant..." or "Here is the 
        summary I re-wrote based on my instructions")
    - Remove subjective terms like "delicious" and keep the description as factual and objective as possible.
    - If the PROJECT DESCRIPTION is shorter than 40 words, just return "Insufficient information"
    - If the PROJECT DESCRIPTION is a single sentence, return "Insufficient information"
    - Do not include any non-UTM symbols in the re-written description, even if they appear in the PROJECT DESCRIPTION

    """

# Function to summarize a single record
def summarize_one(record: dict, model: str = MODEL_NAME) -> dict:
    prompt = format_prompt(record["original_description"], record['program_name'], record["program_context"])
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        summary = response["message"]["content"].strip()
    except Exception as e:
        summary = f"[ERROR] {str(e)}"
    return {'awardee_name': record['grantee_name'],
            'fain': record['fain'],
            'award_amount': record['award_amount'], 
            'original_description': record['original_description'],
            "updated_description": summary}


# Multithreaded batch processor
def summarize_parallel(data: list[dict], model: str = MODEL_NAME, max_workers: int = 4) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(summarize_one, record, model) for record in data]
        for future in tqdm(as_completed(futures)):
            results.append(future.result())
    return results



##########################
'''
PREP THE DATA
Pull the Excel file in with a tab for each program.
Standardize the column names.
Rbind them all into one dataframe
Convert df to dict for the model
'''
##########################

LFPP = pd.read_excel('USASpendingDescriptionsFY22-24.xlsx', sheet_name = 'LFPP')
FMPP = pd.read_excel('USASpendingDescriptionsFY22-24.xlsx', sheet_name = 'FMPP')
RFSP = pd.read_excel('USASpendingDescriptionsFY22-24.xlsx', sheet_name = 'RFSP')
OMDG = pd.read_excel('USASpendingDescriptionsFY22-24.xlsx', sheet_name = 'OMDG')

LFPP = LFPP.rename(columns = {
    'FY': 'fiscal_year',
    'FAIN': 'fain',
    'Organizaton': 'grantee_name',
    'Award Amount': 'award_amount',
    'Published Project Description': 'original_description'
    })
LFPP['program_name'] = 'LFPP'
LFPP['program_context'] = context.lfpp_context

FMPP = FMPP.rename(columns = {
    'FY': 'fiscal_year',
    'FAIN': 'fain',
    'Organizaton': 'grantee_name',
    'Award Amount': 'award_amount',
    'Published Project Description': 'original_description'
    })
FMPP['program_name'] = 'FMPP'
FMPP['program_context'] = context.fmpp_context

RFSP = RFSP.rename(columns = {
    'FY': 'fiscal_year',
    'FAIN': 'fain',
    'Organizaton': 'grantee_name',
    'Award Amount': 'award_amount',
    'Published Project Description': 'original_description'
    })
RFSP['program_name'] = 'RFSP'
RFSP['program_context'] = context.rfsp_context

OMDG['fiscal_year'] = None
OMDG['fain'] = None
OMDG = OMDG[['fiscal_year', 'fain', 'Legal Name', 'Amount Requested', 'Project Summary']]
OMDG = OMDG.rename(columns = {
    'fiscal_year': 'fiscal_year',
    'fain': 'fain',
    'Legal Name': 'grantee_name',
    'Amount Requested': 'award_amount',
    'Project Summary': 'original_description'
    })
OMDG['program_name'] = 'OMDG'
OMDG['program_context'] = context.omdg_context

# combine them into one
d = pd.concat([OMDG, FMPP, LFPP, RFSP])

# convert to dict
d = d.to_dict('records')

start = time.time()
results = summarize_parallel(d, model=MODEL_NAME, max_workers=4)
print(f"\nDone in {time.time() - start:.2f} seconds\n")

# Convert to DataFrame
df = pd.DataFrame(results)
print(df)

# Optional: save to file
df.to_csv("summarized_projects_parallel.csv", index=False)
'''




