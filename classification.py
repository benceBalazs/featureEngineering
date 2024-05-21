import torch
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

with torch.no_grad():
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

    def flatten_array_of_tensors(array_of_tensors):
        flattened_tensors = [torch.flatten(t) for t in array_of_tensors]
        flattened_array = torch.cat(flattened_tensors)
        return flattened_array.numpy()


    # ids = ["VUL4J-1", "VUL4J-3", "VUL4J-4", "VUL4J-5", "VUL4J-6", "VUL4J-7", "VUL4J-8", "VUL4J-10", "VUL4J-12",
    # "VUL4J-18", "VUL4J-19", "VUL4J-20", "VUL4J-22", "VUL4J-23", "VUL4J-25", "VUL4J-26", "VUL4J-30", "VUL4J-39",
    # "VUL4J-40", "VUL4J-41", "VUL4J-43", "VUL4J-44", "VUL4J-46", "VUL4J-47", "VUL4J-50", "VUL4J-53", "VUL4J-55",
    # "VUL4J-57", "VUL4J-59", "VUL4J-61", "VUL4J-64", "VUL4J-65", "VUL4J-66", "VUL4J-73", "VUL4J-74"]
    vulnerable_data = []
    transformed_data = []

    buggy_location = 0
    fixed_location = 0

    vulnerable_features = []
    transformed_features = []

    testvulnerable_line = []

    ids = ["VUL4J-6", "VUL4J-7", "VUL4J-8", "VUL4J-10"]

    max_length = 512
    for ident in ids:
        with open(f"VJBench-trans/{ident}/buggyline_location.json", "r") as file:
            jsondata = json.load(file)
            buggy_location = jsondata["original"][0][0]
            fixed_location = jsondata["rename+code_structure"][0][0]
        with open(f"VJBench-trans/{ident}/{ident}_original_method.java", "r") as file:
            counter = 1
            for line in file:
                if counter == buggy_location:
                    vulnerable_data.append(line.strip())
                counter += 1
        with open(f"VJBench-trans/{ident}/{ident}_full_transformation.java", "r") as file:
            counter = 1
            for line in file:
                if counter == fixed_location:
                    transformed_data.append(line.strip())
                counter += 1

#    print(vulnerable_data)
#    print(transformed_data)

    padding_length_vul = determine_padding_length(vulnerable_data, max_length)
    padding_length_transformed = determine_padding_length(transformed_data, max_length)
    padding_length = max(padding_length_vul, padding_length_transformed)

    processed_vul = preprocess_and_tokenize(vulnerable_data)
    processed_transformed = preprocess_and_tokenize(transformed_data)

    for tokens, ids in processed_vul:
        outputs = model(ids)
        vulnerable_feature = outputs.last_hidden_state
        padded_vectors = torch.nn.functional.pad(vulnerable_feature,
                                                 (0, 0, 0, padding_length - vulnerable_feature.shape[1]), "constant", 1)
        reshaped_vector = padded_vectors.view(1, padding_length, -1)
        vulnerable_features.append((tokens, reshaped_vector))

    for tokens, ids in processed_transformed:
        outputs = model(ids)
        transformed_feature = outputs.last_hidden_state
        padded_vectors = torch.nn.functional.pad(transformed_feature,
                                                 (0, 0, 0, padding_length - transformed_feature.shape[1]), "constant",
                                                 1)
        reshaped_vector = padded_vectors.view(1, padding_length, -1)
        transformed_features.append((tokens, reshaped_vector))

    vulnerable_features_toFlatten = []
    transformed_features_toFlatten = []

    # Vectors printing vulnerable
    for tokens, feature_vector in vulnerable_features:
#        print("=" * 50)
#        print("Tokens:", tokens)
#        print("Feature Vector Shape:", feature_vector.shape)
#        print("Feature Vector:", feature_vector)
#        if len(testvulnerable_line) == 0:
#            testvulnerable_line.append(feature_vector)
        vulnerable_features_toFlatten.append(feature_vector)
#        print("=" * 50)

    # Vectors printing transformed
    for tokens, feature_vector in transformed_features:
#        print("=" * 50)
#        print("Tokens:", tokens)
#        print("Feature Vector Shape:", feature_vector.shape)
#        print("Feature Vector:", feature_vector)
        transformed_features_toFlatten.append(feature_vector)
#        print("=" * 50)

    # Create labels (1 for vulnerable, 0 for transformed)
    labels = np.array([1, 0])

    # flatten features?
#    flattened_array_vul = torch.flatten(torch.stack(vulnerable_features_toFlatten))
#    flattened_array_transformed = torch.flatten(torch.stack(transformed_features_toFlatten))

    flattened_array_vul = flatten_array_of_tensors(vulnerable_features_toFlatten)
    flattened_array_transformed = flatten_array_of_tensors(transformed_features_toFlatten)

    # Combine features and labels
    all_features = np.vstack((flattened_array_vul, flattened_array_transformed))

#    print(all_features)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42)

    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)

    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    print("Test:", rf_classifier.predict(X_test))

    # Evaluate the classifier
    accuracy = rf_classifier.score(X_test, y_test)
    print("Accuracy:", accuracy)

    # import vulnerable and clean
    # create array structure
    # feature extraction
    # combining features and labels
    # training
    # testing
    # confusion matrix?
