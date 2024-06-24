Morphological and Synthetic Analysis assignments

**Language**: English

# HW 1

**Text sources**: the longest novels in English (Ulysses by James Joyce, Infinite Jest by David Foster Wallace, Atlas Shrugged by Ayn Rand, In Search of Lost Time by Marcel Proust), UD English EWT training corpus, and 300,000+ English lexicon.

Original texts were splitte into words, filtered from digits and punctuation, filtered from closed words and words shorter than 2 characters.

Complete lexicon acquired from the source has a size of 297,000+ unique lemmas (for full vocabulary of 380,000+ words), which was cropped to 65,000+ words that occurs more than once. 

UD English EWT training corpus has available tags that were not used in the final version of this work. 

**Language parts**: verbs, nouns, ajectives, adverbs and other open-class parts of speech were covered

During the lexicon preprocessing, the following were done:

- source texts were filtered and concatenated to a single lexicon 
- plural and singular nouns (or 3rd person and initial present verbs) were merged into single data entries
- verb tenses merged into single data entries of their lemmas
- past tense forms of irregular verbs merged to their present tense forms

**The following tags were defined:**
- **NREG** - regular nouns, have forms ending in -s/-es
- **VREG** - regular verbs, have forms ending in -ed and -ing
- **AREG** - regular adjectives, have comparative (-er) and superlative (-est) forms
- **NIRG** - irregular nouns, have uncommon plural form
- **VIRG** - irregular verbs, have uncommon past tense forms
- **AIRG** - irregular adjectives, don't have comp. and super. forms
- **ADV** - adverbs, have forms with and without endings -ly/-wise/-ward

Total number of processed words: **65106**

Total number of lemmas with combined tags: **55372**

Total number of lemmas with separate tags: **63751**

Total number of unique stems: **22998**

Total number of entries per tag:
- NREG: **46904**
- VREG: **9958**
- AREG: **1367**
- NIRG: **26**
- VIRG: **90**
- AIRG: **3471**
- ADV: **1935**

During lexicon acquisition word lemmas were cropped to their stems based on common English prefixes and suffixes, such that each stem ends in consonant. 

**Code launching:**

Install pip package dependencies from [requirements.txt](../requirements.txt) by running:

`python pip install -r requirements.txt
`

but, the following will be sufficient as well:

`python pip install pandas
`

Check existence of closed class words and irregular noun, verb, adjective lists in folder [util_word_lists](util_word_lists)

In case, you want to run the processing from the beginning, make sure folder [tmp_files](../tmp_files) is empty and [lexicon.txt](../lexicon.txt) result file doesn't exists.

Put .txt source files into the folder, then run [main_script.py](../main_script.py) with flag argument --folder_path, like:

`python main_script.py --folder_path folder_name/
`

or use [source_corpora](../source_corpora) folder by default by running:

`python main_script.py
`

After termination of the script resulting lexicon file [lexicon.txt](../lexicon.txt) will be written. 

In addition, folder [tmp_files](../tmp_files) will contain:
- [eng_clean_vocab.txt](tmp_files%2Feng_clean_vocab.txt) English vocabulary containing filtered unique words from source corpora
- [large-lexicon-freq.txt](tmp_files%2Flarge-lexicon-freq.txt) unique lemmas and their frequencies obtained from source corpora
- [lexicon-freq.txt](tmp_files%2Flexicon-freq.txt) common lemmas and their frequencies from the large lexicon
- [large_lexicon.csv](tmp_files%2Flarge_lexicon.csv) table containing lexicon's lemma, stem, tag, count columns, where tags for each lemma are combined into single space-separated string of values

# HW 2

**Covered parts of Speech**: Nouns, Adjectives, Verbs, Adverbs, closed class words

**Lexicon generation**: running [list_fill.py](list_fill.py) will produce separate parts of lexicon for different parts of speech and save them to the [util_lex_parts](util_lex_parts) directory (based on the words lists and previously acquired lexicon [lexicon.txt](util_word_lists%2Flexicon.txt) stored in the [util_word_lists](util_word_lists) directory)

Before running the python script to generate partial lexicon files, make sure pandas library is installed:

`pip install pandas
`

Then you can run:

`python list_fill.py
`

This should generate separate .lexc files for nouns, regular and irregular verbs, adjectives, adverbs, and closed class words.

**NOTE!** The main lexicon file [lex.lexc](util_lex_parts%2Flex.lexc) should be defined manually. It contains rules for prefixes, suffixes, and endings of all POS. 

Next step should be a concatenation of the generated lexicons into a single file using list of lexicon part files [lexlist.txt](lexlist.txt):

`{ xargs cat < lexlist.txt ; } > english.lexc
`

**Foma compilation**: next step is loading the grammar from [grammar.foma](grammar.foma) file that requires [english.lexc](english.lexc) full lexicon file:

`foma
`

`source grammar.foma
`

Grammar is ready to use and can be saved to the launchable [english.bin](english.bin) file:

`save stack english.bin
`

**Testing**: edit [input.txt](input.txt) file containing words to look up in the defined grammar (list of 40 samples is defined by default).

Make sure [english.bin](english.bin) exists in the current directory.

Then run [script.sh](script.sh) to go through the list of words in the input file:

`./script.sh
`

Manually defined parts of the lexicon can be found in [grammar.foma](grammar.foma) and [lex.lexc](util_lex_parts%2Flex.lexc) files. 
