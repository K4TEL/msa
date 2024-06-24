import string
from collections import Counter
import pandas as pd
import re
import os
import copy


closed_file_path = "util_word_lists/closed_class_list.txt"

irg_verbs_file_path = "util_word_lists/irregular_verbs_list.txt"
irg_nouns_file_path = "util_word_lists/irregular_nouns_list.txt"
irg_adjs_file_path = "util_word_lists/irregular_adj_list.txt"

eng_clean_vocab_file_path = "tmp_files/eng_clean_vocab.txt"

punctuation = '"!#$%&()*+,./:;<=>?@[\]^_`{|}~®¬“…—-‘”'
vowels = "aeiouAEIOU"

predicted_cols = ["STEM", "TAG"]
result_cols = ['LEMMA', 'STEM', 'TAG']


noun_plur_spec_rules = {
    "us": "i",
    "is": "es",
    "um": "a",
    "ex": "ices",
}


# V to V
prefix_verb = ["re", "dis", "over", "un", "mis", "out", "be", "co", "de", "fore", "inter", "pre", "sub", "trans",
               "under", "em", "en", "an"]
suffix_verb = ["ise", "ize", "ate", "fy", "ify", "ed", "ing"]

# N to N
prefix_noun = ["anti", "auto", "bi", "co", "counter", "dis", "ex", "hyper", "in", "inter", "kilo", "mal", "mega",
               "mis", "mini", "mono", "neo", "out", "poly", "pseudo", "re", "semi", "sub", "super", "sur", "tele",
               "tri", "ultra", "under", "vice", "mid", "pre", "over", "an", "multi", "non", "quad", "oct", "fore",
               "trans", "di", "dia", "il", "im", "ir", "bio", "micro", "uni", "ante", "com", "con", "pro", "intra",
               "hypo", "omni", "homo", "hetero"]

# V to N
suffix_noun_v = ["tion", "sion", "ment", "ent", "ant", "al", "ence", "ance", "erely", "ry", "ure",
                 "ion", "ation", "ition", "ee"]
# A to N
suffix_noun_a = ["ity", "ty", "ness", "cy", "dom", "ism"]
# N to N
suffix_noun_n = ["er", "or", "ism", "ship", "age", "ery", "hood", "ist", "ess", "ian", "an", "ary", "tude"]

# A to A
prefix_adj = ["un", "non", "dis", "ir", "im", "in", "il", "ab", "a", "post"]
# A to A
suffix_adj_a = ["est", "er"]
# V, N to A
suffix_adj_nv = ["al", "ant", "ent", "ive", "ous", "ful", "less", "able", "ible", "ic", "ish", "y", "some", "esque",
              "ative", "itive", "eous", "ious", "en", "ial"]


suffix_adv = ["ly", "wise", "ward"]

tag_suf = {
    "VREG": suffix_verb,
    "AREG": suffix_adj_a,
    "ADV": suffix_adv,
    "AIRG": suffix_adj_nv,
    "NREG": suffix_noun_v + suffix_noun_a + suffix_noun_n
}

suffixes = list(set(suffix_verb + suffix_adj_a + suffix_adv +
                    suffix_adj_nv + suffix_noun_v + suffix_noun_a + suffix_noun_n))
for s in ["cy", "ry", "ty", "y"]:
    suffixes.remove(s)
suffixes = sorted(suffixes, key=lambda x: len(x), reverse=True)


prefixes = list(set(prefix_noun + prefix_adj + prefix_verb))
prefixes = sorted(prefixes, key=lambda x: len(x), reverse=True)


class LexParser:
    def __init__(self,):
        self.text_sources = None
        self.eng_vocab = None
        self.lexicon = None
        self.lex_table = None
        self.result = None

        with open(closed_file_path, 'r') as input:
            text = input.read()
        self.closed_words = text.lower().split("\n")

        with open(irg_verbs_file_path, 'r') as input:
            text = input.read()
        self.irg_verbs = text.lower().split("\n")

        with open(irg_nouns_file_path, 'r') as input:
            text = input.read()
        self.irg_nouns = text.lower().split("\n")

        with open(irg_adjs_file_path, 'r') as input:
            text = input.read()
        self.irg_adjs = text.lower().split("\n")

    def source_list(self, folder):
        text_sources = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        print(f'Files in {folder}:')
        res = []
        for file_name in text_sources:
            print(file_name)
            res.append(folder + file_name)

        self.text_sources = res

        return res

    def sources_to_lexicon(self, output_file, sources=None):
        if os.path.exists(output_file):
            print(f'File {output_file} already exists')
            lex = pd.read_csv(output_file, sep="\t")
            print(f"Size:\t{len(lex.index)}")
            if os.path.exists(eng_clean_vocab_file_path):
                print(f'File {eng_clean_vocab_file_path} already exists')
                with open(eng_clean_vocab_file_path, 'r') as input:
                    text = input.read()
                self.eng_vocab = text.lower().split("\n")
                print(f"Size:\t{len(self.eng_vocab)}")
        else:
            words = []
            text_sources = sources if sources else self.text_sources

            for source in text_sources:
                text = txt_to_words(source)
                words += text

            words = filter_words(words, self.closed_words)
            print(f"Total text words count\t{len(words)}")
            words_count = Counter(words)
            print(f"Total unique words count\t{len(words_count.keys())}")

            with open(eng_clean_vocab_file_path, 'w') as out:
                for word in words_count.keys():
                    out.write(f"{word}\n")
            print(f"{len(words_count.keys())} clean words saved to {eng_clean_vocab_file_path}")

            self.eng_vocab = list(words_count.keys())

            merged_lex = merge_singular_and_plural(words_count, self.closed_words)
            print(f"Merged plur-singular words count\t{len(merged_lex.keys())}")
            merged_lex = merge_verb_tenses(merged_lex, self.irg_verbs, self.closed_words)
            print(f"Merged verb tense words count\t{len(merged_lex.keys())}")

            lex = pd.DataFrame.from_dict(merged_lex, orient='index', columns=['cnt']).reset_index()
            lex.columns = ['LEMMA', 'cnt']

            lex = lex.sort_values(by='cnt', ascending=False)
            lex = lex.reset_index(drop=True)

            # print(lex)
            # print(lex.info())

            lex[["LEMMA", "cnt"]].to_csv(output_file, sep='\t', index=False)
            print(f"{len(lex.index)} lemmas saved to {output_file}")

        return lex

    def common_lexicon(self, input_file, output_file, dataset_size=None):
        if os.path.exists(output_file):
            print(f'File {output_file} already exists')
            lex = pd.read_csv(output_file, sep="\t")
            lex.dropna(inplace=True)
            print(f"Size:\t{len(lex.index)}")
        else:
            lexicon = pd.read_csv(input_file, sep='\t')

            # print(lex)
            # print(lexicon.info())

            if dataset_size:
                lex = lexicon.head(dataset_size)
            else:
                lex = lexicon[lexicon["cnt"] > 1]

            lex[["LEMMA", "cnt"]].to_csv(output_file, sep='\t', index=False)
            print(f"{len(lex.index)} common lemmas saved to {output_file}")

            lex = pd.read_csv(output_file, sep='\t')
            lex.dropna(inplace=True)

            # print(lex.info())

        self.lexicon = lex

        return lex

    def parse(self, table_output_file):
        if os.path.exists(table_output_file):
            print(f'File {table_output_file} already exists')
            lex_table = pd.read_csv(table_output_file, sep=",")
            tagged = lex_table.dropna()
            print(f"Size:\t{len(lex_table.index)}")
            print(f"Tagged:\t{len(tagged.index)}")
        else:
            print(f'Processing lexicon of size {len(self.lexicon.index)}...')
            lex_table = self.lexicon
            lex_table[predicted_cols] = lex_table['LEMMA'].apply(lambda x: pd.Series(self.get_stem_tag(x)))
            #
            # print(lex)
            print(lex_table.info())
            #
            lex_table.to_csv(table_output_file, sep=",", index=False)
            print(f"{len(lex_table.index)} parsing results saved to {table_output_file}")

        self.lex_table = lex_table

        return lex_table

    def get_stem_tag(self, word):
        word_dict = {
            "LEMMA": word,
            "STEM": [],
            "TAG": [],
            "pref": [],
            "suf": []
        }

        for irg_vrb in self.irg_verbs:
            irg_vrb = irg_vrb.replace("\\", "-")
            init_verb = irg_vrb.split("-")[0]
            if word == init_verb:
                return word, "VIRG"

        for irg_adj in self.irg_adjs:
            irg_adj = irg_adj.replace("\\", "-")
            adjs = irg_adj.split("-")
            if word in adjs:
                return word, "AIRG"

        for irg_nn in self.irg_nouns:
            irg_nn = irg_nn.replace("\\", "-")
            nouns = irg_nn.split("-")
            if word in nouns:
                return word, "NIRG"

        if vocab_check_adv(word, self.eng_vocab):
            word_dict["TAG"].append("ADV")
            for suf in suffix_adv:
                if word.endswith(suf):
                    word_dict["STEM"].append(f"{word[:-len(suf)]}")
                    word_dict["suf"].append(suf)

        if vocab_check_reg_plur(word, self.eng_vocab):
            word_dict["TAG"].append("NREG")
            word_dict["STEM"].append(word)

        if vocab_check_spc_plur(word, self.eng_vocab):
            word_dict["TAG"].append("NSPC")
            for sing_end, plur_end in noun_plur_spec_rules.items():
                if word.endswith(sing_end):
                    word_dict["STEM"].append(f"{word[:-len(sing_end)]}")
                    word_dict["suf"].append(sing_end)
                elif word.endswith(plur_end):
                    word_dict["STEM"].append(f"{word[:-len(plur_end)]}")
                    word_dict["suf"].append(plur_end)

        if vocab_check_adj_degree(word, self.eng_vocab):
            word_dict["TAG"].append("AREG")
            stem = word
            if word.endswith("e") and len(word[:-1]) > 2:
                stem = stem[:-1]
                word_dict["suf"].append("e")
            elif word.endswith("er") and len(word[:-2]) > 2:
                stem = stem[:-2]
                word_dict["suf"].append("er")
            elif len(word) > 3 and word.endswith("est") and len(word[:-3]) > 2:
                stem = stem[:-3]
                word_dict["suf"].append("est")
            word_dict["STEM"].append(stem)

        if vocab_check_verb_tenses(word, self.eng_vocab):
            word_dict["TAG"].append("VREG")
            stem = word
            for suf in ["e", "y"]:
                if word.endswith(suf) and len(word[:-len(suf)]) > 2:
                    stem = word[:-1]
                    word_dict["suf"].append(suf)
            word_dict["STEM"].append(stem)

        found_dict = copy.deepcopy(word_dict)
        word_dict = unite_result(word_dict)

        match = False
        for tag, suf_list in tag_suf.items():
            for suf in suf_list:
                if word.endswith(suf):
                    stem = word[:-len(suf)]
                    if len(stem) > 2:

                        if word_dict["TAG"] is not None:
                            if tag in word_dict["TAG"]:
                                # print("MATCH", stem, tag, suf)

                                if word_dict["suf"] is not None:
                                    if len(suf) > len(word_dict["suf"]):
                                        word_dict["STEM"] = crop_double_end(stem)
                                        word_dict["suf"] = [suf]
                                        # print(word_dict)
                                else:
                                    # print(word_dict)
                                    # word_dict["TAG"] = tag
                                    word_dict["STEM"] = crop_double_end(stem)
                                    word_dict["suf"] = [suf]
                                    # print(word_dict)
                                match = True
                            elif tag not in word_dict["TAG"] and not match:
                                # print(word_dict)
                                # print("DIFF", word, stem, tag, suf)
                                found_dict["STEM"].append(stem)
                                found_dict["TAG"].append(tag)
                                found_dict["suf"].append(suf)

                        else:
                            found_dict["STEM"].append(stem)
                            found_dict["TAG"].append(tag)
                            found_dict["suf"].append(suf)

        if word_dict["TAG"] is None:
            if len(found_dict["TAG"]) == 0:
                # print("--UNMATCH--", word)
                word_dict["STEM"] = word
            else:
                word_dict = unite_result(found_dict, True)
                # print("--FOUND--", found_dict)
                # print("\t", word_dict["TAG"])

        crop = True
        while any(word_dict["STEM"].startswith(pref) for pref in prefixes) and crop:
            word_dict, crop = crop_pref(word_dict, self.eng_vocab)
            # if crop:
            #     print("pref", crop, word_dict)

        crop = True
        while any(word_dict["STEM"].endswith(suf) for suf in suffixes) and crop:
            word_dict, crop = crop_suf(word_dict, self.eng_vocab)
            # if crop:
            #     print("suf", crop, word_dict)

        # print(word_dict)
        word_dict["STEM"] = word_dict["STEM"][:-1] if vowel_end(word_dict["STEM"]) and len(
            word_dict["STEM"][:-1]) > 2 else word_dict["STEM"]

        return word_dict["STEM"], word_dict["TAG"]

    def lex_save(self, output_file):
        if os.path.exists(output_file):
            print(f'File {output_file} already exists')
            split_df = pd.read_csv(output_file, sep="\t")

            print(f"Lexicon size:\t{len(split_df.index)}")

            tag_counts = split_df["TAG"].value_counts()
            print(f"Lexicon split tags counts:")
            print(tag_counts)
            stem_counts = split_df["STEM"].value_counts()
            print(f"Lexicon unique stems count:\t{len(stem_counts[stem_counts == 1])}")

        else:
            man_df = self.lex_table[result_cols].dropna()
            print(f"Lexicon size:\t{len(man_df)}")

            tag_counts = man_df["TAG"].value_counts()
            print(f"Lexicon tags counts:")
            print(tag_counts)
            stem_counts = man_df["STEM"].value_counts()
            print(f"Lexicon unique stems count:\t{len(stem_counts[stem_counts == 1])}")

            split_df = man_df.apply(split_tags, axis=1).explode('TAG').reset_index(drop=True)

            tag_counts = split_df["TAG"].value_counts()
            print(f"Lexicon split tags counts:")
            print(tag_counts)

            split_df.to_csv(output_file, sep='\t', index=False)
            print(f"{len(split_df.index)} entries saved to {output_file}")

        self.result = split_df

        return split_df


def filter_words(words_list, closed_words):
    result = []
    for word in words_list:
        word = str(word).lower()

        if len(word) < 2:
            continue

        if word in closed_words:
            continue

        if any(char.isdigit() for char in word):
            continue

        if all(char in string.punctuation for char in word):
            continue

        if any(char in "’'-" for char in word):
            continue

        if re.search(r'(.)\1\1', word):
            continue

        cleaned_word = ''.join([' ' + char + ' ' if char in punctuation else char for char in word])
        result.extend(cleaned_word.split())

    words = [item for item in result if not any(char in string.punctuation for char in item)]

    return words


def txt_to_words(file_path):
    with open(file_path, 'r') as input:
        text = input.read()
    return text.split("\n")


def merge_verb_tenses(dictionary, irg_verbs, closed_words):
    merged_dict = {}

    for word, count in dictionary.items():

        reg_form = None

        for irg_vrb in irg_verbs:
            irg_vrb = irg_vrb.replace("\\", "-")
            verbs = irg_vrb.split("-")
            if word in verbs:
                reg_form = verbs[0]

                # print("irg", reg_form, word, irg_vrb)

                if reg_form == word:
                    merged_dict[word] = count
                elif reg_form in dictionary.keys():
                    merged_dict[reg_form] = dictionary[reg_form] + count
                elif reg_form not in dictionary.keys():
                    merged_dict[reg_form] = count

        if not reg_form:

            if word.endswith("ing"):
                lem = word[:-3]
                if len(lem) > 1:
                    reg_form = lem
                    # print("ING", reg_form, word)

            elif word.endswith("ed"):
                lem = word[:-2]
                if len(lem) > 2:
                    reg_form = lem[:-1] + "y" if lem.endswith("i") else lem
                    # print("ED", reg_form, word)

            else:
                # print("\tnon", word)
                merged_dict[word] = count

        if reg_form and reg_form not in closed_words:

            if reg_form in dictionary.keys():
                # print("found", reg_form, word)
                merged_dict[reg_form] = dictionary[reg_form] + count

            elif reg_form not in dictionary.keys():
                preg_form = reg_form
                for p in prefix_verb:
                    if reg_form.startswith(p):
                        stemmed = reg_form[len(p):]
                        preg_form = stemmed if len(stemmed) > 2 else reg_form

                prege_form = preg_form + "e"
                rege_form = reg_form + "e"

                regg_form = crop_double_end(reg_form)
                pregg_form = crop_double_end(preg_form)

                found_modification = False
                for mod in [preg_form, pregg_form, prege_form, rege_form, regg_form]:
                    if mod in dictionary.keys():
                        merged_dict[mod] = dictionary[mod] + count
                        found_modification = True

                if not found_modification:
                    saved = False
                    for s in ["iz", "is", "at", "ar"]:
                        if reg_form.endswith(s):
                            merged_dict[rege_form] = count
                            saved = True

                    if word.endswith("eed"):
                        comp = True
                        for cons in ["c", "r", "t", "k"]:
                            if ends_with(cons, word[:-3]):
                                merged_dict[rege_form] = count
                                comp = False
                                saved = True

                        if comp:
                            merged_dict[word] = count
                            saved = True

                    if not saved:
                        merged_dict[word] = count

    return merged_dict


def merge_singular_and_plural(dictionary, closed_words):
    merged_dict = {}

    for word, count in dictionary.items():

        singular_forms = []
        sing_form = None

        strict_pass = False

        if word.endswith('s') or word.endswith('es'):

            if word.endswith('s') and len(word) > 3:
                if word[-2] not in vowels and word[-2] != "s":
                    sing_form = [word[:-1]]
                elif word.endswith("eaus"):
                    sing_form = [word[:-1]]
                elif word.endswith("sis"):
                    sing_form = [word]
                elif word[-2] in vowels and word[-2] not in ["e", "u"]:
                    sing_form = [word[:-1]]

                if sing_form:
                    singular_forms += sing_form

            if word.endswith('es') and len(word) > 2:
                if word[-3] == "i":
                    sing_form = [word[:-3] + "y", word[:-1]]
                elif word[-3] == "v":
                    sing_form = [word[:-3] + "fe", word[:-3] + "f", word[:-1]]
                elif any(word[:-2].endswith(end) for end in ["x", "ch", "sh"]):
                    sing_form = [word[:-2]]
                elif any(word[:-2].endswith(end) for end in ["zz", "ss"]):
                    sing_form = [word[:-3], word[:-2]]
                elif word[:-2].endswith("s"):
                    sing_form = [word[:-2] + "is", word[:-2], word[:-1]]
                elif word[:-2].endswith("o"):
                    sing_form = [word[:-1], word[:-2]]
                elif word[:-2].endswith("ic"):
                    sing_form = [word[:-4] + "ex", word[:-1], word[:-2]]
                else:
                    sing_form = [word[:-1]] if len(word[:-1]) > 2 else None

                if sing_form:
                    singular_forms += sing_form

            if len(singular_forms) > 0:
                found_sing = False
                for singular in singular_forms:
                    if singular in dictionary.keys() and not found_sing:
                        if singular not in closed_words:
                            merged_dict[singular] = dictionary[singular] + count
                        found_sing = True

                if not found_sing:
                    if len(singular_forms) == 1:
                        for suf in suffixes:
                            if singular_forms[0].endswith(suf):
                                strict_pass = True

                    if strict_pass:
                        merged_dict[singular_forms[0]] = count
                    else:
                        merged_dict[word] = count
            else:
                merged_dict[word] = count

        else:
            merged_dict[word] = count

    return merged_dict


def vowel_end(word):
    return word[-1] in vowels


def cons_end(word):
    return word[-1] not in vowels


def crop_double_end(stem):
    return stem[:-1] if stem[-1] == stem[-2] else stem


def ends_with(char, word):
    return len(word) > 0 and word[-1] == char


def vocab_check_reg_plur(word, eng_vocab):
    plur_forms = []

    if any(word.endswith(end) for end in ["ss", "x", "ch", "sh"]):
        plur_forms.append(f"{word}es")

    elif word.endswith("o"):
        plur_forms.append(f"{word}s")
        plur_forms.append(f"{word}es")

    elif any(word.endswith(end) for end in ["s", "z"]):
        plur_forms.append(f"{word}{word[-1]}es")
        plur_forms.append(f"{word}es")

    elif word.endswith("y") and vowel_end(word[:-1]):
        plur_forms.append(f"{word}s")
    elif word.endswith("y") and cons_end(word[:-1]):
        plur_forms.append(f"{word[:-1]}ies")

    elif any(word.endswith(end) for end in ["fe", "f"]):
        plur_forms.append(f"{word[:-2]}ves")
        plur_forms.append(f"{word[:-1]}ves")

    else:
        plur_forms.append(f"{word}s")

    # print(word, plur_forms)
    return any(plural in eng_vocab for plural in plur_forms)


def vocab_check_adj_degree(word, eng_vocab):
    degrees = [word]
    if word.endswith("er"):
        degrees.append(f"{word[:-2]}")
        degrees.append(f"{word[:-1]}st")
    elif word.endswith("est"):
        degrees.append(f"{word[:-2]}r")
        degrees.append(f"{word[:-3]}")
    elif word.endswith("e"):
        degrees.append(f"{word}st")
        degrees.append(f"{word}r")
    else:
        degrees.append(f"{word}est")
        degrees.append(f"{word}er")

    # print(degrees)
    return all(degree in eng_vocab for degree in degrees)


def vocab_check_verb_tenses(word, eng_vocab):
    tenses = [word, word + "ing"]
    if word.endswith("y"):
        tenses.append(f"{word[:-1]}ied")
    elif word.endswith("e"):
        tenses.append(f"{word}d")
    elif cons_end(word):
        tenses.append(f"{word}ed")
        tenses.append(f"{word}{word[-1]}ed")
        tenses.append(f"{word}{word[-1]}ing")
    else:
        tenses.append(f"{word}ed")

    # print(tenses)
    common_elements = [tense for tense in tenses if tense in eng_vocab]
    # print(common_elements, tenses)
    return len(common_elements) >= 3


def vocab_check_spc_plur(word, eng_vocab):
    word_forms = [word]
    for sing_end, plur_end in noun_plur_spec_rules.items():
        if word.endswith(sing_end):
            word_forms.append(f"{word[:-len(sing_end)]}{plur_end}")
        elif word.endswith(plur_end):
            word_forms.append(f"{word[:-len(plur_end)]}{sing_end}")
        else:
            return False

    # print(word_forms)
    return all(form in eng_vocab for form in word_forms)


def vocab_check_adv(word, eng_vocab):
    forms = [word]
    for suf in suffix_adv:
        if word.endswith(suf) and len(word[:-len(suf)]) > 2:
            forms.append(f"{word[:-len(suf)]}")
    if len(forms) < 2:
        return False

    # print(forms)
    return all(form in eng_vocab for form in forms)


def crop_pref(word_dict, eng_vocab):
    cropped = False
    for pref in prefixes:
        if word_dict["STEM"].startswith(pref):
            stem = word_dict["STEM"][len(pref):]
            if len(stem) > 2:
                if word_dict["LEMMA"][len(pref):] in eng_vocab:
                    word_dict["STEM"] = stem
                    word_dict["pref"].append(pref)
                    cropped = True

    return word_dict, cropped


def crop_suf(word_dict, eng_vocab):
    cropped = False
    for suf in suffixes:
        if word_dict["STEM"].endswith(suf):
            stem = word_dict["STEM"][:-len(suf)]
            if len(stem) > 2:
                if word_dict["LEMMA"][:-len(suf)] in eng_vocab:
                    s = crop_double_end(stem)
                    word_dict["STEM"] = s if len(s) > 2 else stem
                    word_dict["suf"].append(suf)
                    cropped = True

    return word_dict, cropped


def unite_result(word_dict, min_stem=False):
    if len(word_dict["TAG"]) == 1:
        word_dict["STEM"] = word_dict["STEM"][0]
        word_dict["TAG"] = word_dict["TAG"][0]
    elif len(word_dict["TAG"]) > 1:
        stem_set = list(set(word_dict["STEM"]))
        if min_stem and len(stem_set) == len(word_dict["STEM"]):
            stems = word_dict["STEM"]
            ind = stems.index(min(stems, key=len))

            word_dict["TAG"] = word_dict["TAG"][ind]
            word_dict["STEM"] = word_dict["STEM"][ind]
            word_dict["suf"] = [word_dict["suf"][ind]]

        else:
            word_dict["TAG"] = " ".join(list(set(word_dict["TAG"])))
            word_dict["STEM"] = min(list(set(word_dict["STEM"])), key=len)
            word_dict["suf"] = list(set(word_dict["suf"]))

        word_dict["pref"] = list(set(word_dict["pref"]))
    else:
        word_dict["STEM"] = None
        word_dict["TAG"] = None

    return word_dict


def split_tags(row):
    tags = row['TAG'].split()
    return pd.Series({'LEMMA': row['LEMMA'], 'STEM': row['STEM'], 'TAG': tags})