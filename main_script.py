from lex_parser import *
import argparse

large_lemma_freq_file = 'tmp_files/large-lexicon-freq.txt'
word_freq_file = 'tmp_files/lexicon-freq.txt'
lexicon_table = "tmp_files/large_lexicon.csv"

lexicon_result = "lexicon.txt"

sources_folder_path = 'source_corpora/'


def main():
    parser = argparse.ArgumentParser(description='List files in a folder.')
    parser.add_argument('--folder_path', type=str, default=sources_folder_path, help='Path to the folder (default: current directory)')
    args = parser.parse_args()

    lex_parser = LexParser()

    lex_parser.source_list(args.folder_path)

    print(f"Parsing source corpora to lemma-freq file {large_lemma_freq_file}")
    lex_parser.sources_to_lexicon(large_lemma_freq_file)

    print(f"Cropping lexicon to the most common lemmas in file {word_freq_file}")
    lex_parser.common_lexicon(large_lemma_freq_file, word_freq_file)

    print(f"Acquisition of stems and tags for lemmas")
    lex_parser.parse(lexicon_table)

    print(f"Writing split tags lexicon to file {lexicon_result}")
    lex_parser.lex_save(lexicon_result)


if __name__ == '__main__':
    main()