import pickle
import os
import json

# Constants
DATA_DIRS = [
    "data/raw/Amazon/",
    "data/raw/TripAdvisor/",
    "data/raw/Yelp/",
]
DESTINATION_DIRS = [
    "data/preprocessed/Amazon/",
    "data/preprocessed/TripAdvisor/",
    "data/preprocessed/Yelp/",
]


def load_ids(data_dir):
    """Load and return IDs and ID to exp mapping."""
    with open(os.path.join(data_dir, "id2exp.json"), "r", encoding="utf-8") as f:
        id2exp = json.load(f)
    with open(os.path.join(data_dir, "IDs.pickle"), "rb") as f:
        IDs = pickle.load(f)
    return IDs, id2exp


def generate_index_maps(IDs, id2exp):
    """Generate and return user, item, and exp index mappings."""
    user_set, item_set, exp_set = set(), set(), set()
    for record in IDs:
        user_set.add(record["user"])
        item_set.add(record["item"])
        exp_set |= set(record["exp_idx"])

    user_list = list(user_set)
    item_list = list(item_set)
    exp_list = list(exp_set)
    text_list = [id2exp[e] for e in exp_list]
    return {
        "user2index": {x: i for i, x in enumerate(user_list)},
        "item2index": {x: i for i, x in enumerate(item_list)},
        "exp2index": {x: i for i, x in enumerate(exp_list)},
        "text_list": text_list,
        "exp_list": exp_list,
    }


def format_data(data_dir, data_type, partition, indexes_map):
    """Format data based on data_type and return as a list of tuples."""
    with open(os.path.join(data_dir, partition, data_type + ".index"), "r") as f:
        line = f.readline()
        indexes = [int(x) for x in line.split(" ")]

    tuple_list = []
    for idx in indexes:
        record = IDs[idx]
        u = indexes_map["user2index"][record["user"]]
        i = indexes_map["item2index"][record["item"]]
        exp_list = record["exp_idx"]
        exps = list(set([indexes_map["exp2index"][e] for e in exp_list]))
        texts = list(set([id2exp[e] for e in exp_list]))
        if "test" in data_type:
            tuple_list.append([u, i, exps, texts])
        else:
            for ex, tx in zip(exps, texts):
                tuple_list.append([u, i, ex, tx, exps, texts])
    return tuple_list


def write_file(name, file_content):
    """Write file_content to a file named name."""
    with open(name, "w") as fp:
        for item in file_content:
            fp.write(f"{item}\n")


def write_json(name, content):
    """Write JSON content to a file."""
    with open(name, "w") as convert_file:
        convert_file.write(json.dumps(content))


# Main logic
for origin_data, destination in zip(DATA_DIRS, DESTINATION_DIRS):
    origin_data = os.path.join("..", origin_data)
    destination = os.path.join("..", destination)
    IDs, id2exp = load_ids(origin_data)
    indexes_map = generate_index_maps(IDs, id2exp)

    for partition in range(1, 6):
        for dtype in ["train", "test"]:
            data_content = format_data(origin_data, dtype, f"{partition}/", indexes_map)
            file_name = os.path.join(destination, f"{dtype}{partition}.txt")
            dir_name = os.path.dirname(file_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            write_file(file_name, data_content)

    # Write index maps and lists to JSON
    write_json(f"{destination}user2index.txt", indexes_map["user2index"])
    write_json(f"{destination}item2index.txt", indexes_map["item2index"])
    write_json(f"{destination}exp2index.txt", indexes_map["exp2index"])
    write_json(f"{destination}text_list.txt", indexes_map["text_list"])
    write_json(f"{destination}exp_list.txt", indexes_map["exp_list"])
