from os.path import exists, join
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dir', type=str, required=True)

args = parser.parse_args()

assert exists(args.dir)

print("\n\n\n")

if exists(join(args.dir, "eval_textvqa.log")):
    with open(join(args.dir, "eval_textvqa.log")) as f:
        lines = f.readlines()
    acc = None
    for line in lines:
        if line.startswith("Accuracy:"):
            acc = round(float(line.split()[-1][:-1]), 1)
            break
    assert acc is not None
    print("TextVQA:", acc)

if exists(join(args.dir, "eval_mme.log")):
    with open(join(args.dir, "eval_mme.log")) as f:
        lines = f.readlines()
    acc = None
    for line in lines:
        if line.startswith("total score: "):
            acc = round(float(line.split()[-1]))
            break
    assert acc is not None
    print("MME:", acc)

if exists(join(args.dir, "eval_sqa.log")):
    with open(join(args.dir, "eval_sqa.log")) as f:
        lines = f.readlines()
    acc = None
    for line in lines:
        if "IMG-Accuracy" in line:
            acc = round(float(line.split()[-1][:-1]), 1)
            break
    assert acc is not None
    print("SQA:", acc)

if exists(join(args.dir, "eval_gqa.log")):
    with open(join(args.dir, "eval_gqa.log")) as f:
        lines = f.readlines()
    acc = None
    for line in lines:
        if line.startswith("Accuracy:"):
            acc = round(float(line.split()[-1][:-1]), 1)
            break
    assert acc is not None
    print("GQA:", acc)

if exists(join(args.dir, "eval_seed.log")):
    with open(join(args.dir, "eval_seed.log")) as f:
        lines = f.readlines()
    acc = None
    for line in lines:
        if line.startswith("image accuracy:"):
            acc = round(float(line.split()[-1][:-1]), 1)
            break
    assert acc is not None
    print("SEED-I:", acc)

if exists(join(args.dir, "eval_pope.log")):
    with open(join(args.dir, "eval_pope.log")) as f:
        lines = f.readlines()
    acc = []
    for line in lines:
        if line.startswith("F1 score:"):
            acc.append(round(float(line.split()[-1][:-1]) * 100, 1))
    
    assert len(acc) > 0
    acc = round(sum(acc)/len(acc), 1)
    print("POPE:", acc)



print("\n")
print(args.dir.split("/")[-2])