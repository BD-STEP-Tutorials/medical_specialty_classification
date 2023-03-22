import argparse
import pandas as pd
from pathlib import Path
import spacy
import en_core_sci_sm


def parse_args():
    """ Get command line arguments """

    parser = argparse.ArgumentParser(description="Cleans input csv file")

    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i',
                        '--input',
                        help='Path to input file',
                        type=Path,      # Note that this requires import as follows “from pathlib import Path”
                        default=None,
                        required=True)

    # optional (keyword) argument with '-o' flag to specify path to output file(s) that will be generated
    parser.add_argument('-o',
                        '--output',
                        help='Path to output file',
                        type=Path,
                        default=None,
                        required=True)

    args = parser.parse_args()
    return args

def clean_text(nlp, text: str) -> str:
    ''' Clean text using passed spacy language model'''

    doc = nlp(text_lower)

    # lowercase string
    text_lower = text.lower()
    
    # extract text if the token contains all letters and is not a stop work or punctuation mark
    tokens = [token.text for token in doc if (
                                                (token.text.isalpha() == True) & 
                                                (token.is_stop == False) & 
                                                (token.is_punct == False)
                                                )]    

    # return extracted tokens as a joined string
    return " ".join(tokens)    

def extract_entities(nlp, doc: str) -> str:
    ''' Extracts the text for entities identified in doc using passed spacy language model'''

    doc = nlp(doc)
    ent_list = [ent.text for ent in doc.ents]

    # return text for extracted entities as a joined string
    return " ".join(ent_list)


def main():

    args = parse_args()

    text_df = pd.read_csv(args.input)

    # load spacy (scispacy) model
    nlp = en_core_sci_sm.load()

    # clean transcription texts and store into new col
    text_df['tx_clean'] = text_df['transcription'].apply(nlp, clean_text)

    # clean transcription texts and store into new col
    text_df['tx_clean_ents'] = text_df['tx_clean'].apply(nlp, extract_entities)
    
    # write df with added text cols from processing to file 
    text_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()