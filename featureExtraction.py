import os
import torch
from transformers import AutoTokenizer, AutoModel

# Load CodeBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
model = AutoModel.from_pretrained('microsoft/codebert-base')


# Define function to load Java code
def load_data(ident):
    with open(f"VJBench-trans/{ident}/{ident}_original_method.java", "r") as file:
        code = file.read()
    return code


# Define function to preprocess and tokenize Java code
def preprocess_and_tokenize(code):
    return_tokens = tokenizer.tokenize(code)
    input_ids = tokenizer.encode(code, return_tensors="pt")
    return return_tokens, input_ids


# Define function to bucket/bin data based on sequence length
def bucket_data(data):
    buckets = {}
    for java_code in data:
        tokens, input_ids = preprocess_and_tokenize(java_code)
        length = input_ids.size(1)
        max_length = min(length, model.config.max_position_embeddings)
        if max_length not in buckets:
            buckets[max_length] = []
        buckets[max_length].append((tokens, input_ids))
    return buckets


# Define function to pad/truncate sequences within each bucket/bin
def pad_or_truncate(given_buckets, given_target_length):
    return_padded_data = []
    for max_length, bucket in buckets.items():
        for tokens, input_ids in bucket:
            if input_ids.size(1) <= max_length:
                padded_input_ids = torch.nn.functional.pad(input_ids, (0, max_length - input_ids.size(1)), "constant",
                                                           0)
            else:
                padded_input_ids = input_ids[:, :max_length]
            return_padded_data.append((tokens, padded_input_ids))
    return return_padded_data


# Define constant length for bucketing
max_length = 512
target_length = 256
max_line = ""
# Load and preprocess data for each id
data = []
# ids = ["VUL4J-1", "VUL4J-3", "VUL4J-4", "VUL4J-5", "VUL4J-6", "VUL4J-7", "VUL4J-8", "VUL4J-10", "VUL4J-12", "VUL4J-18",
#        "VUL4J-19", "VUL4J-20", "VUL4J-22", "VUL4J-23", "VUL4J-25", "VUL4J-26", "VUL4J-30", "VUL4J-39", "VUL4J-40",
#        "VUL4J-41", "VUL4J-43", "VUL4J-44", "VUL4J-46", "VUL4J-47", "VUL4J-50", "VUL4J-53", "VUL4J-55", "VUL4J-57",
#        "VUL4J-59", "VUL4J-61", "VUL4J-64", "VUL4J-65", "VUL4J-66", "VUL4J-73", "VUL4J-74"]

ids = ["VUL4J-3", "VUL4J-4", "VUL4J-46"]
for ident in ids:
    java_code_lines = []
    with open(f"VJBench-trans/{ident}/{ident}_original_method.java", "r") as file:
        for line in file:
            data.append(line.strip())
    java_code = '\n'.join(java_code_lines)
    # data.append(java_code)
print(max_line)
print(len(max_line))
# Bucket/bin data based on sequence length
buckets = bucket_data(data)

# Pad/truncate sequences within each bucket/bin
padded_data = pad_or_truncate(buckets, target_length)

# Reshape padded sequences and get contextual vectors
contextual_vectors = []
for tokens, padded_input_ids in padded_data:
    outputs = model(padded_input_ids)
    contextual_vector = outputs.last_hidden_state.squeeze(0)
    contextual_vector = contextual_vector.unsqueeze(0)
    contextual_vectors.append((tokens, contextual_vector))

# Example of using contextual vectors
for tokens, contextual_vector in contextual_vectors:
    print("=" * 50)
    print("Tokens:", tokens)
    print("Contextual Vector Shape:", contextual_vector.shape)
    print("Contextual Vector:", contextual_vector)
    print("=" * 50)
