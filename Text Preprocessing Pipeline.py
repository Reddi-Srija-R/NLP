import spacy
from langdetect import detect
import re
import os
import logging

logging.basicConfig(filename='preprocessing.log', level=logging.INFO)


CONTRACTION_MAP = { "ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "could've": "could have", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'll": "I will", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would",
    "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "must've": "must have",
    "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not", "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there'd": "there would", "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what're": "what are", "what's": "what is", "what've": "what have", "where'd": "where did", "where's": "where is", "who'll": "who will", "who's": "who is",
    "won't": "will not", "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have" }


# Function to create a customizable stop_words list
def customize_stopwords(additional_stopwords = None, remove_stopwords = None):
    # call this function after lang_detect instead of loading it here
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    default_stopwords = set(nlp.Defaults.stop_words)
    custom_stopwords = default_stopwords.copy()

    if additional_stopwords:
        custom_stopwords.update(additional_stopwords)
        logging.info("Stopwords are updated.")

    if remove_stopwords:
        custom_stopwords -= set(remove_stopwords)
        logging.info("Specified stopwords are removed.")

    return custom_stopwords

# Function to detect language
# Using 'langdetect' (as 'langid' took 3x time to execute the code)
def detect_lang(text):
    try:
        lang = detect(text)
        logging.info("Language has been detected.")
    except:
        lang = 'unknown'
    return lang

# Function to load spacy model corresponding to the language detected
def load_spacy_model(lang_detected):

    if lang_detected == 'en':
        # !python -m spacy download en_core_web_sm
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        logging.info("English language has been loaded.")
        return nlp
    
    elif lang_detected == 'fr':
        # !python -m spacy download fr_core_news_sm
        nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])
        logging.info("French language has been loaded.")
        return nlp
    
    elif lang_detected == 'es':
        # !python -m spacy download es_core_news_sm
        nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
        logging.info("Spanish language has been loaded.")
        return nlp
    
    elif lang_detected == 'de':
        # !python -m spacy download de_core_news_sm
        nlp = spacy.load('de_core_news_sm', disable=['parser', 'ner'])
        logging.info("Germany language has been loaded.")
        return nlp
    
    else:
        return

# Function to tokenize text
def spacy_tokenize(nlp, text, do_tokenize):

    if do_tokenize == 'yes':
        doc = nlp(text)
        tokens = [token.text for token in doc]
        logging.info("Tokenization is performed.")
        if tokens is not None:
            return tokens
    else:
        return text.split()
 
# Function to remove stop words
def remove_stopwords(nlp, tokens, custom_stopwords = None):

    if custom_stopwords:
        doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
        tokens = [token for token in tokens if token.lower() not in custom_stopwords]
        logging.info("Stopwords are removed.")
    else:
        doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
        tokens = [token.text for token in doc if not token.is_stop]
        logging.info("Stopwords are removed.")
    return tokens
 
# Function to apply lemmatization
def apply_lemmatization(nlp, tokens, do_lemmatize):

    if do_lemmatize == 'yes':
        doc = nlp(' '.join(tokens))  
        tokens = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc]
        logging.info("Lem matization is performed.")
    return tokens

# Function to expand contractions
def expand_contractions(text, contraction_map):

    pattern = re.compile('({})'.format('|'.join(contraction_map.keys())),
                         flags=re.IGNORECASE | re.DOTALL)
    
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_map.get(match) if contraction_map.get(match) else contraction_map.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    
    expanded_text = pattern.sub(expand_match, text)
    logging.info("Contractions are expanded.")
    return expanded_text
 
# Function to manage negations (handling 'not' and 'no')
def manage_negations(text):
    
    # Replace 'not' and 'no' followed by a token with 'not_' + token or 'no_' + token
    negation_pattern = re.compile(r'\b(not|no)\s+(?=\w)', flags=re.IGNORECASE)
    text = negation_pattern.sub(lambda x: x.group(0).replace(' ', '_'), text)
    logging.info("Negations are managed.")
    return text
 
# Function to handle special characters and emojis
def handle_special_characters(text, do_special_characters):

    if do_special_characters == 'yes':
        # Remove special characters except spaces and alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Handle emojis
        text = text.encode('ascii', 'ignore').decode('ascii')
        logging.info("Special characters and emojis are handled.")
    return text
 
# Main preprocessing function that applies steps based on user options
def preprocess_text(nlp, text, tokenize='no', stopwords='no', lemmatize='no', special_characters='no', custom_stopwords = None):
    
    tokens = spacy_tokenize(nlp, text, tokenize)
    tokens = remove_stopwords(nlp, tokens, stopwords)
    tokens = apply_lemmatization(nlp, tokens, lemmatize)
    cleaned_text = ' '.join(tokens)
    cleaned_text = expand_contractions(cleaned_text, CONTRACTION_MAP) 
    cleaned_text = manage_negations(cleaned_text)  
    cleaned_text = handle_special_characters(cleaned_text, special_characters) 
    return cleaned_text
  
# Function to preprocess a single file
def preprocess_file(input_file, output_folder, tokenize='no', stopwords='no', lemmatize='no', special_characters='no', custom_stopwords = None):

    try:
        output_file = os.path.join(output_folder, os.path.basename(input_file))

        # Read text from input file
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

            segments = text.split('\n')

            for s in segments:
                lang = detect_lang(s)
                nlp = load_spacy_model(lang)
                
                if nlp is not None:
                    # Preprocess text based on user options
                    cleaned_text = preprocess_text(nlp, text, tokenize, stopwords, lemmatize, special_characters, custom_stopwords)

                    # Save cleaned text to output file
                    with open(output_file, 'w', encoding='utf-8') as file:
                        file.write(cleaned_text)
                    
                    logging.info("Preprocessing of a input file is completed.")
            print(f'Processed file: {input_file} -> {output_file}')
    
    except FileNotFoundError as F:
        logging.error(F)
 
# Function to preprocess all files in a folder
def preprocess_folder(input_folder, output_folder, tokenize='no', stopwords='no', lemmatize='no', special_characters='no', custom_stopwords = None):

    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Iterate through files in input folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):  
                input_file = os.path.join(input_folder, filename)
                preprocess_file(input_file, output_folder, tokenize, stopwords, lemmatize, special_characters)
        
        logging.info("Preprocessing of whole input data is completed.")
        print('Preprocessing complete.')
    
    except FileNotFoundError as F:
        logging.error(F)

# Main function
input_folder = 'Extracted Output Folder'   
output_folder = 'Processed Output Folder'  

# Specify preprocessing options ('yes' or 'no' for each step)
tokenize = input("Enter 'yes' if you would want to tokenize:")
stopwords = input("Enter 'yes' if you would want to remove stopwords:")
lemmatize = input("Enter 'yes' if you would want to apply lemmatization:")
special_characters = input("Enter 'yes' if you would want to handle special characters:")

'''custom stop words configuration'''
# additional_stopwords = ['custom', 'list']
# remove_stopwords_list = ['some','words']
# custom_stopwords = customize_stopwords(additional_stopwords, remove_stopwords_list)

# Perform preprocessing on all files in the input folder
preprocess_folder(input_folder, output_folder, tokenize, stopwords, lemmatize, special_characters, custom_stopwords = None)