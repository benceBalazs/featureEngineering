import torch
from transformers import AutoTokenizer, AutoModel

# Load CodeBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
model = AutoModel.from_pretrained('microsoft/codebert-base')


# Load Java code
def load_data(inner_ident):
    with open(f"VJBench-trans/{inner_ident}/{inner_ident}_original_method.java", "r") as open_file:
        code = open_file.read()
    return code


# Preprocess and tokenize Java code
def preprocess_and_tokenize(code):
    return_items = []
    for inner_line in code:
        inner_tokens = tokenizer.tokenize(inner_line)
        inner_input_ids = tokenizer.encode(inner_line, return_tensors="pt")
        return_items.append((inner_tokens, inner_input_ids))
    return return_items


# Dynamic padding/truncation
def determine_padding_length(given_data, given_max_length):
    # Initialize a list to store sequence lengths
    sequence_lengths = []

    # Tokenize each line and store its length
    for inner_java_code_lines in given_data:
        inner_input_ids = tokenizer.encode(inner_java_code_lines, return_tensors="pt")
        length = inner_input_ids.size(1)
        sequence_lengths.append(length)

    # Determine padding length as the maximum sequence length or max_length, whichever is smaller
    inner_padding_length = min(max(sequence_lengths), given_max_length)

    return inner_padding_length


# ids = ["VUL4J-1", "VUL4J-3", "VUL4J-4", "VUL4J-5", "VUL4J-6", "VUL4J-7", "VUL4J-8", "VUL4J-10", "VUL4J-12",
# "VUL4J-18", "VUL4J-19", "VUL4J-20", "VUL4J-22", "VUL4J-23", "VUL4J-25", "VUL4J-26", "VUL4J-30", "VUL4J-39",
# "VUL4J-40", "VUL4J-41", "VUL4J-43", "VUL4J-44", "VUL4J-46", "VUL4J-47", "VUL4J-50", "VUL4J-53", "VUL4J-55",
# "VUL4J-57", "VUL4J-59", "VUL4J-61", "VUL4J-64", "VUL4J-65", "VUL4J-66", "VUL4J-73", "VUL4J-74"]
data = []
feature_vectors = []
ids = ["VUL4J-6"]
max_length = 512
for ident in ids:
    java_code_lines = []
    with open(f"VJBench-trans/{ident}/{ident}_original_method.java", "r") as file:
        for line in file:
            data.append(line.strip())

padding_length = determine_padding_length(data, max_length)

processed = preprocess_and_tokenize(data)

for tokens, ids in processed:
    # Pass tokens through CodeBERT model
    outputs = model(ids)
    feature_vector = outputs.last_hidden_state
    #    print("Contextual Vectors Shape:", feature_vector.shape)

    # Padding or truncation to achieve constant length
    padded_vectors = torch.nn.functional.pad(feature_vector, (0, 0, 0, padding_length - feature_vector.shape[1]), "constant", 0)
    #    print("Padded Vectors Shape:", padded_vectors.shape)

    # Reshape to desired shape
    reshaped_vector = padded_vectors.view(1, padding_length, -1)
    #    print(reshaped_vector)
    #    print("Reshaped Vectors Shape:", reshaped_vector.shape)

    feature_vectors.append((tokens, reshaped_vector))

# Vectors printing
for tokens, feature_vector in feature_vectors:
    print("=" * 50)
    print("Tokens:", tokens)
    print("Feature Vector Shape:", feature_vector.shape)
    print("Feature Vector:", feature_vector)
    print("=" * 50)
