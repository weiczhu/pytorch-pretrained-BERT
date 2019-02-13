import random
import hashlib

def build_dataset(mode):
    data = {}
    with open("glue_data/JAPC/intent_{}.csv".format(mode), encoding="utf-8") as file:
        for line in file:
            if line.strip():
                label = line.strip().split(" ")[0]
                sentence = line.strip().replace(label + " ", "")
                if label not in data:
                    data[label] = []
                data[label].append(sentence)

    data_pair = []
    keys = data.keys()
    for key in keys:
        sentences = data[key]
        for s in sentences:
            c1 = random.choice(list(filter(lambda x: x != s, sentences)))
            positive_pair = ("1", s, c1)

            other_key = random.choice(list(filter(lambda x: x != key, keys)))
            c2 = random.choice(data[other_key])
            negative_pair = ("0", s, c2)

            data_pair.extend([positive_pair, negative_pair])

    lines = ["Quality	#1 ID	#2 ID	#1 String	#2 String"]
    for label, s, c in data_pair:
        s_id = sentence_to_id(s)
        c_id = sentence_to_id(c)

        line = "\t".join([label, s_id, c_id, s, c])
        lines.append(line)

    with open("glue_data/JAPC/{}.tsv".format(mode), 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l + "\n")


def sentence_to_id(sentence):
    return str(int(hashlib.md5(sentence.encode()).hexdigest(), 16))[-7:]


if __name__ == "__main__":
    print("enter main")
    build_dataset("train")
    build_dataset("test")
    print("exit main")

