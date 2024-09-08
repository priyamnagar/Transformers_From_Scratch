# Python Libraries
import torch
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from torch.utils.data import DataLoader, TensorDataset

class DataPreprocessing():
    def __init__(self,
                 data_path,
                 max_sentence_length,
                 batch_size):
        super().__init__()
        self.data_path = data_path
        self.max_sentence_length = max_sentence_length
        self.batch_size = batch_size
        self.english = None
        self.french = None
        self.english_int_word = {}
        self.english_word_int = {}
        self.english_vocab = None
        self.french_vocab = None
        self.french_int_word = {}
        self.french_word_int = {}
        self.dataset = None
        self.dataloader = None
        self.english_vocab_size = None
        self.french_vocab_size = None

    def load_data(self):
        data = pd.read_csv(self.data_path, encoding='utf-8')
        self.english = data["English words/sentences"].tolist()
        self.french = data["French words/sentences"].tolist()

    def get_vocab(self, sentences):
        tokenizer = RegexpTokenizer(r'\w+')
        vocab = []
        vocab.append("<START>")
        vocab.append("<END>")
        vocab.append("<PADDING>")
        for i in sentences:
            for j in tokenizer.tokenize(i):
                vocab.append(j.lower())
        vocab = list(set(vocab))

        int_to_word = {i: word for i, word in enumerate(vocab, 1)}  # Starts numbering from 1
        word_to_int = {word: i for i, word in int_to_word.items()}  # Reverse dictionary
        print(max(int_to_word.keys()))
        
        return vocab, int_to_word, word_to_int

    def convert_sentences_integers(self, sentences, word_int, int_word):
        tokenizer = RegexpTokenizer(r'\w+')
        sentences_int = []
        for i in sentences:
            sentences_sent = [word_int["<START>"]]
            for j in tokenizer.tokenize(i):
                sentences_sent.append(word_int[j.lower()])
            if(len(sentences_sent) >= self.max_sentence_length):
                sentences_sent = sentences_sent[:self.max_sentence_length-1]
                sentences_sent.append(word_int["<END>"])
            elif(len(sentences_sent) < self.max_sentence_length):
                sentences_sent.append(word_int["<END>"])
                for _ in range(self.max_sentence_length - len(sentences_sent)):
                    sentences_sent.append(word_int["<PADDING>"])
            
            if(len(sentences_sent) !=self.max_sentence_length):
                print("Found error in length: \n")
                print(len(sentences_sent))
                    
            sentences_int.append(sentences_sent)

        return sentences_int

    def prepare_dataset(self, input_data, output_data):
        input_data = torch.Tensor(input_data)
        output_data = torch.Tensor(output_data)
        dataset = TensorDataset(input_data.long(), output_data.long())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataset, dataloader
    
    def forward(self):
        print(f"Loading Data from {self.data_path}")
        self.load_data()

        print("Creating English vocabulary...")
        self.english_vocab, self.english_int_word, self.english_word_int = self.get_vocab(self.english)
        self.english_vocab_size = len(self.english_vocab)

        print("Creating French vocabulary...")
        self.french_vocab, self.french_int_word, self.french_word_int = self.get_vocab(self.french)
        self.french_vocab_size = len(self.french_vocab)
        
        print("Converting English Sentences to integer...")
        self.english_int = self.convert_sentences_integers(self.english, self.english_word_int, self.english_int_word)

        print("Converting French Sentences to integer...")
        self.french_int = self.convert_sentences_integers(self.french, self.french_word_int, self.french_int_word)

        print("Preparing Dataset for Input...")
        self.dataset, self.dataloader = self.prepare_dataset(self.english_int, self.french_int)

        print("Done.")
        
        return self.dataset, self.dataloader