import pandas as pd

irg_verbs_file_path = "util_word_lists/irregular_verbs_list.txt"
irg_nouns_file_path = "util_word_lists/irregular_nouns_list.txt"
irg_adjs_file_path = "util_word_lists/irregular_adj_list.txt"
closed_file_path = "util_word_lists/closed_class_list.txt"


with open(closed_file_path, 'r') as input:
    text = input.read()
closed = text.lower().split("\n")[:-1]

c = ["LEXICON Closed"]
for i, w in enumerate(closed):
    c.append(f"\n{w}\t#;")

with open("util_lex_parts/cls.lexc", "w") as myfile:
    for l in c:
        myfile.write(l)


with open(irg_verbs_file_path, 'r') as input:
    text = input.read()
irg_verbs = text.lower().split("\n")[:-1]

print(irg_verbs)

u1, u2, u3, = ["\n\nLEXICON V1unique"], ["\n\nLEXICON V2unique"], ["\n\nLEXICON V3unique"]

for v in irg_verbs:
    forms = v.split("-")
    # print(forms)

    if len(set(forms)) == 1:
        w1 = forms[0]
        u1.append(f"\n{w1}\tVREnd;")
        u1.append(f"\n{w1}+V+PS:{w1}\t#;")
        u1.append(f"\n{w1}+V+PP:{w1}\t#;")
        u1.append(f"\n{w1}+V+FP:{w1}\t#;")
    elif len(set(forms)) == 2:
        w1, w2 = forms[0], forms[1]
        u2.append(f"\n{w1}\tVREnd;")
        u2.append(f"\n{w1}+V+PS:{w2}\t#;")
        u2.append(f"\n{w1}+V+PP:{w2}\t#;")
        u2.append(f"\n{w1}+V+FP:{w2}\t#;")
    else:
        w1 = forms[0]
        w2 = forms[1].split("\\")
        w3 = forms[2].split("\\")

        u3.append(f"\n{w1}\tVREnd;")
        for f in w2:
            u3.append(f"\n{w1}+V+PS:{f}\t#;")
        for f in w3:
            u3.append(f"\n{w1}+V+PP:{f}\t#;")
            u3.append(f"\n{w1}+V+FP:{f}\t#;")

with open("util_lex_parts/virg.lexc", "a") as myfile:
    for l in u1:
        myfile.write(l)
    for l in u2:
        myfile.write(l)
    for l in u3:
        myfile.write(l)



lex = "util_word_lists/lexicon.txt"

df = pd.read_csv(lex, sep="\t")
print(df.info())

advs = df[df["TAG"] == "ADV"]
print(advs.info())

nouns = df[df["TAG"] == "NREG"]
print(nouns.info())

adjs = df[df["TAG"] == "AREG"]
print(adjs.info())

verbs = df[df["TAG"] == "VREG"]
print(verbs.info())

verbs_list = verbs["LEMMA"].tolist()
print(verbs_list)

v = ["LEXICON VREG", "\nNoun\tNVSuf;", "\nAdj\tAVSuf;"]
for w in verbs_list:
    v.append(f"\n{w}\tVEnd;")

with open("../util_lex_parts/verg.lexc", "w") as myfile:
    for l in v:
        myfile.write(l)


adj_list = adjs["STEM"].tolist()
print(adj_list)

a = ["LEXICON AREG"]
for w in adj_list:
    a.append(f"\n{w}\tAEnd;")

with open("util_lex_parts/areg.lexc", "w") as myfile:
    for l in a:
        myfile.write(l)


noun_list = nouns["LEMMA"].tolist()
stem_list = nouns["STEM"].tolist()
print(noun_list)
print(stem_list)

n = ["LEXICON NREG", "\nVerb\tVNSuf;", "\nAdj\tANSuf;"]
for i, w in enumerate(noun_list):
    n.append(f"\n{w}\tNEnd;")
    n.append(f"\n{stem_list[i]}\tNSuf;")

with open("util_lex_parts/nreg.lexc", "w") as myfile:
    for l in n:
        myfile.write(l)

advs_list = advs["LEMMA"].tolist()
print(advs_list)

ad = ["LEXICON ADV", "\nNoun\tADVSuf;", "\nAdj\tADVSuf;"]
for i, w in enumerate(advs_list):
    ad.append(f"\n{w}\t#;")

with open("util_lex_parts/adv.lexc", "w") as myfile:
    for l in ad:
        myfile.write(l)

