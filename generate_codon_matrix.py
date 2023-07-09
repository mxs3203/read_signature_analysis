
import glob
import os

import pandas as pd
import re
import tqdm as tqdm
import pickle as pk


def generate_strings(vocabulary = "ACTG", max_length = 3):
    strings = []

    # Recursive function to generate strings
    def generate(prefix, length):
        if length == 0:
            strings.append(prefix)
            return
        for char in vocabulary:
            generate(prefix + char, length - 1)

    # Generate strings of length 1, 2, and 3
    for length in range(1, min(max_length + 1, len(vocabulary) + 1)):
        generate("", length)

    return strings

def count_string_occurrences(string, substrings, count):
    res = {}
    for substring in substrings:
        res[substring] = len(re.findall('(?={})'.format(substring), string)) * count/len(string)
    return res

all_codons = generate_strings()
df = pd.read_csv("/Users/au589901/Desktop/Pager/vidjil_nt_2022Apr/3867_2.tsv", sep="\t")
example_string = df.loc[0, 'CDR3.nt']
result = count_string_occurrences(example_string, all_codons, count=1)
assert result['A'] == 5*1/len(example_string)
assert result['TT'] == 4*1/len(example_string)
assert result['CGT'] == 1*1/len(example_string)
example_string = df.loc[30, 'CDR3.nt']
result = count_string_occurrences(example_string, all_codons, count=30)
assert result['C'] == 6*30/len(example_string)
assert result['TT'] == 3*30/len(example_string)
assert result['GGG'] == 4*30/len(example_string)
files = glob.glob("/Users/au589901/Desktop/Pager/vidjil_nt_2022Apr/*.tsv")


for f in tqdm.tqdm(files):
    df = pd.read_csv(f, sep="\t")
    id = f.split('/')[-1].split('.')[0] # current ID
    if not os.path.exists("data/{}_freq.csv".format(id)):
        res = df.apply(lambda row: count_string_occurrences(row['CDR3.nt'], all_codons, 1), axis=1).to_list()
        df_dictionary = pd.DataFrame(res)
        df_dictionary['id'] = id
        df_dictionary['Num_Clones'] = df['Clones']
        #total_res = pd.concat([total_res, df_dictionary], ignore_index=True)
        #with open("data/{}_freq.csv", 'w+b') as p:
        df_dictionary.to_csv("data/{}_freq.csv".format(id))
        #pk.dump(total_res, p, pk.HIGHEST_PROTOCOL)
    else:
        print("Already exists... skipping {}".format(id))