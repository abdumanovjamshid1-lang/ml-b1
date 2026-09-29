import math

# -------------------------------------------------
# 1. TRAINING DATA
# -------------------------------------------------

data = [
    {"Parcel": "P1",  "Traffic": "Low",    "Weather": "Clear", "Courier Load": "Light", "Distance": "Short", "Target": "ON_TIME"},
    {"Parcel": "P2",  "Traffic": "Medium", "Weather": "Clear", "Courier Load": "Light", "Distance": "Short", "Target": "ON_TIME"},
    {"Parcel": "P3",  "Traffic": "Low",    "Weather": "Clear", "Courier Load": "Light", "Distance": "Long",  "Target": "ON_TIME"},
    {"Parcel": "P4",  "Traffic": "Low",    "Weather": "Rain",  "Courier Load": "Light", "Distance": "Short", "Target": "ON_TIME"},
    {"Parcel": "P5",  "Traffic": "Medium", "Weather": "Clear", "Courier Load": "Light", "Distance": "Long",  "Target": "ON_TIME"},
    {"Parcel": "P6",  "Traffic": "Medium", "Weather": "Rain",  "Courier Load": "Light", "Distance": "Short", "Target": "DELAYED"},
    {"Parcel": "P7",  "Traffic": "Medium", "Weather": "Clear", "Courier Load": "Heavy", "Distance": "Long",  "Target": "DELAYED"},
    {"Parcel": "P8",  "Traffic": "High",   "Weather": "Clear", "Courier Load": "Light", "Distance": "Long",  "Target": "DELAYED"},
    {"Parcel": "P9",  "Traffic": "Low",    "Weather": "Rain",  "Courier Load": "Heavy", "Distance": "Long",  "Target": "DELAYED"},
    {"Parcel": "P10", "Traffic": "High",   "Weather": "Clear", "Courier Load": "Heavy", "Distance": "Short", "Target": "DELAYED"},
    {"Parcel": "P11", "Traffic": "High",   "Weather": "Rain",  "Courier Load": "Heavy", "Distance": "Long",  "Target": "SEVERE_DELAY"},
    {"Parcel": "P12", "Traffic": "High",   "Weather": "Rain",  "Courier Load": "Heavy", "Distance": "Short", "Target": "SEVERE_DELAY"},
    {"Parcel": "P13", "Traffic": "Medium", "Weather": "Rain",  "Courier Load": "Heavy", "Distance": "Long",  "Target": "SEVERE_DELAY"},
    {"Parcel": "P14", "Traffic": "High",   "Weather": "Rain",  "Courier Load": "Light", "Distance": "Long",  "Target": "SEVERE_DELAY"},
    {"Parcel": "P15", "Traffic": "High",   "Weather": "Clear", "Courier Load": "Heavy", "Distance": "Long",  "Target": "SEVERE_DELAY"}
]

features = ["Traffic", "Weather", "Courier Load", "Distance"]
classes = ["ON_TIME", "DELAYED", "SEVERE_DELAY"]


# -------------------------------------------------
# 2. ENTROPY
# H(D) = -sum(p_k * log2(p_k))
# -------------------------------------------------

def entropy(rows):
    counts = {}

    for row in rows:
        label = row["Target"]

        if label not in counts:
            counts[label] = 0

        counts[label] += 1

    total = len(rows)
    result = 0.0

    for label in counts:
        p = counts[label] / total
        result -= p * math.log2(p)

    return result


# -------------------------------------------------
# 3. SPLIT DATA BY A FEATURE
# Example: split_by_feature(data, "Weather")
# -------------------------------------------------

def split_by_feature(rows, feature):
    groups = {}

    for row in rows:
        value = row[feature]

        if value not in groups:
            groups[value] = []

        groups[value].append(row)

    return groups


# -------------------------------------------------
# 4. WEIGHTED ENTROPY AND INFORMATION GAIN
# IG(D, A) = H(D) - H(D | A)
# -------------------------------------------------

def weighted_entropy(rows, feature):
    groups = split_by_feature(rows, feature)
    total = len(rows)
    result = 0.0

    for value in groups:
        subset = groups[value]
        weight = len(subset) / total
        result += weight * entropy(subset)

    return result


def information_gain(rows, feature):
    parent_entropy = entropy(rows)
    child_entropy = weighted_entropy(rows, feature)
    return parent_entropy - child_entropy


# -------------------------------------------------
# 5. MAJORITY CLASS
# Used if no features remain
# -------------------------------------------------

def majority_class(rows):
    counts = {}

    for row in rows:
        label = row["Target"]

        if label not in counts:
            counts[label] = 0

        counts[label] += 1

    best_class = None
    best_count = -1

    # Deterministic tie rule:
    # classes list order is used if counts are equal.
    for label in classes:
        if label in counts and counts[label] > best_count:
            best_class = label
            best_count = counts[label]

    return best_class


# -------------------------------------------------
# 6. FIND BEST FEATURE
# In a tie, the earlier feature in the features list wins.
# -------------------------------------------------

def best_feature(rows, available_features):
    best = available_features[0]
    best_gain = information_gain(rows, best)

    for feature in available_features[1:]:
        gain = information_gain(rows, feature)

        if gain > best_gain:
            best_gain = gain
            best = feature

    return best


# -------------------------------------------------
# 7. BUILD ID3 TREE RECURSIVELY
#
# Tree node format:
# {
#   "feature": selected feature,
#   "branches": {...},
#   "default": majority class
# }
#
# A leaf is simply a string, such as "DELAYED".
# -------------------------------------------------

def build_id3_tree(rows, available_features):
    labels = []

    for row in rows:
        labels.append(row["Target"])

    # Stop if all target labels are the same
    all_same = True

    for label in labels:
        if label != labels[0]:
            all_same = False
            break

    if all_same:
        return labels[0]

    # Stop if no unused feature remains
    if len(available_features) == 0:
        return majority_class(rows)

    # Choose feature with largest Information Gain
    selected_feature = best_feature(rows, available_features)

    tree = {
        "feature": selected_feature,
        "branches": {},
        "default": majority_class(rows)
    }

    groups = split_by_feature(rows, selected_feature)

    # Remove selected feature for child nodes
    remaining_features = []

    for feature in available_features:
        if feature != selected_feature:
            remaining_features.append(feature)

    # Build one child branch for every value
    for value in groups:
        subset = groups[value]
        tree["branches"][value] = build_id3_tree(subset, remaining_features)

    return tree


# -------------------------------------------------
# 8. PRINT TREE IN READABLE FORMAT
# -------------------------------------------------

def print_tree(tree, indent=""):
    if type(tree) == str:
        print(indent + "=> " + tree)
        return

    print(indent + "[" + tree["feature"] + "]")

    for value in tree["branches"]:
        print(indent + "  " + value + ":")

        print_tree(tree["branches"][value], indent + "    ")


# -------------------------------------------------
# 9. PREDICTION AND DECISION PATH
# -------------------------------------------------

def predict(tree, sample):
    current_node = tree
    path = []

    while type(current_node) != str:
        feature = current_node["feature"]
        value = sample[feature]

        path.append(feature + " = " + value)

        if value in current_node["branches"]:
            current_node = current_node["branches"][value]
        else:
            # Used only if a totally new/unseen value appears
            current_node = current_node["default"]

    return current_node, path


# -------------------------------------------------
# TASK 1: ENTROPY AND IG(D, Weather)
# -------------------------------------------------

print("========== TASK 1 ==========")
print("Root entropy H(D):", round(entropy(data), 6))
print("Weighted entropy H(D | Weather):", round(weighted_entropy(data, "Weather"), 6))
print("Information Gain IG(D, Weather):", round(information_gain(data, "Weather"), 6))
print()


# -------------------------------------------------
# TASK 2: INFORMATION GAIN FOR ALL FEATURES
# -------------------------------------------------

print("========== TASK 2 ==========")
print("Feature               Information Gain")

for feature in features:
    gain = information_gain(data, feature)
    print(feature.ljust(22), round(gain, 6))

print()


# -------------------------------------------------
# TASK 3: BUILD AND PRINT TREE
# -------------------------------------------------

tree = build_id3_tree(data, features)

print("========== TASK 3 ==========")
print("Learned ID3 tree:")
print_tree(tree)
print()


# -------------------------------------------------
# TASK 4: TEST DATA AND PREDICTION
# Actual Class is kept only for evaluation later.
# -------------------------------------------------

test_data = [
    {"Parcel": "T1", "Traffic": "Low",    "Weather": "Clear", "Courier Load": "Heavy", "Distance": "Short", "Actual": "ON_TIME"},
    {"Parcel": "T2", "Traffic": "High",   "Weather": "Rain",  "Courier Load": "Light", "Distance": "Short", "Actual": "DELAYED"},
    {"Parcel": "T3", "Traffic": "Medium", "Weather": "Rain",  "Courier Load": "Heavy", "Distance": "Short", "Actual": "DELAYED"},
    {"Parcel": "T4", "Traffic": "High",   "Weather": "Clear", "Courier Load": "Light", "Distance": "Short", "Actual": "DELAYED"},
    {"Parcel": "T5", "Traffic": "Medium", "Weather": "Clear", "Courier Load": "Heavy", "Distance": "Short", "Actual": "DELAYED"},
    {"Parcel": "T6", "Traffic": "High",   "Weather": "Rain",  "Courier Load": "Heavy", "Distance": "Long",  "Actual": "SEVERE_DELAY"}
]

print("========== TASK 4 ==========")

predictions = []

for sample in test_data:
    predicted, path = predict(tree, sample)
    predictions.append(predicted)

    print(sample["Parcel"])
    print("  Prediction:", predicted)
    print("  Path:", " -> ".join(path))
    print()


# -------------------------------------------------
# TASK 5: CONFUSION MATRIX AND ACCURACY
# Rows = actual, columns = predicted
# -------------------------------------------------

matrix = {}

for actual in classes:
    matrix[actual] = {}

    for predicted in classes:
        matrix[actual][predicted] = 0

for i in range(len(test_data)):
    actual = test_data[i]["Actual"]
    predicted = predictions[i]

    matrix[actual][predicted] += 1

print("========== TASK 5 ==========")
print("Confusion Matrix")
print("Actual \\ Predicted".ljust(20), "ON_TIME".ljust(15), "DELAYED".ljust(15), "SEVERE_DELAY")

for actual in classes:
    print(
        actual.ljust(20),
        str(matrix[actual]["ON_TIME"]).ljust(15),
        str(matrix[actual]["DELAYED"]).ljust(15),
        str(matrix[actual]["SEVERE_DELAY"]).ljust(15)
    )

correct = 0

for label in classes:
    correct += matrix[label][label]

accuracy = correct / len(test_data)

print()
print("Correct predictions:", correct)
print("Total test samples:", len(test_data))
print("Accuracy:", round(accuracy * 100, 2), "%")


# Find the most common wrong prediction
largest_error = 0
most_confused_actual = ""
most_confused_predicted = ""

for actual in classes:
    for predicted in classes:
        if actual != predicted:
            if matrix[actual][predicted] > largest_error:
                largest_error = matrix[actual][predicted]
                most_confused_actual = actual
                most_confused_predicted = predicted

print("Most common confusion:", most_confused_actual, "predicted as", most_confused_predicted)