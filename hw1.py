import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator


# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    padded = ['<s>'] * (n - 1) + list(text) + ['</s>']
    for i in range(n - 1, len(padded)):
        word = padded[i]
        context = tuple(padded[i - (n - 1):i])
        yield (word, context)


# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = text.split('\n\n')
    sentences = []
    for para in paragraphs:
        if not para.strip():
            continue
        for sent_str in sent_tokenize(para):
            words = word_tokenize(sent_str)
            sentences.append(words)
    return sentences


# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    sentences = load_corpus(corpus_path)
    lm = NGramLM(n)
    for sent in sentences:
        lm.update(sent)
    return lm


# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        for word, context in get_ngrams(self.n, text):
            key = (word, context)
            self.ngram_counts[key] = self.ngram_counts.get(key, 0) + 1
            self.context_counts[context] = self.context_counts.get(context, 0) + 1
            if word != '<s>':
                self.vocabulary.add(word)
            for w in context:
                if w != '<s>':
                    self.vocabulary.add(w)
            if word == '</s>':
                self.vocabulary.add(word)

    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta= .0) -> float:
        v = len(self.vocabulary)
        if context not in self.context_counts:
            return 1.0 / v if v > 0 else 0.0
        if delta == 0:
            count_wc = self.ngram_counts.get((word, context), 0)
            count_c = self.context_counts[context]
            return count_wc / count_c if count_c > 0 else 0.0
        count_wc = self.ngram_counts.get((word, context), 0)
        count_c = self.context_counts[context]
        return (count_wc + delta) / (count_c + delta * v)

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        total = 0.0
        for word, context in get_ngrams(self.n, sent):
            p = self.get_ngram_prob(word, context, delta)
            if p <= 0:
                return -float('inf')
            total += math.log(p, 2)
        return total

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # delta is a float
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]], delta=0.) -> float:
        total_log_prob = sum(self.get_sent_log_prob(sent, delta) for sent in corpus)
        total_tokens = sum(len(sent) for sent in corpus)
        if total_tokens == 0:
            return 0.0
        avg_log_prob = total_log_prob / total_tokens
        return math.pow(2, -avg_log_prob)

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        words = sorted(self.vocabulary.union({'</s>'}))
        r = random.random()
        cumul = 0.0
        for w in words:
            cumul += self.get_ngram_prob(w, context, delta)
            if r < cumul:
                return w
        return words[-1]

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        context = ('<s>',) * (self.n - 1)
        words = []
        for _ in range(max_length):
            word = self.generate_random_word(context, delta)
            if word == '</s>':
                break
            words.append(word)
            context = context[1:] + (word,)
        return ' '.join(words)


def main(corpus_path: str, delta: float, seed: int):
    trigram_lm = create_ngram_lm(3, corpus_path)
    s1 = 'God has given it to me, let him who touches it beware!'
    s2 = 'Where is the prince, my Dauphin?'

    print(trigram_lm.get_sent_log_prob(word_tokenize(s1), delta))
    print(trigram_lm.get_sent_log_prob(word_tokenize(s2), delta))

    gen = trigram_lm.generate_random_text(15, delta)
    print('Generated (%d words):' % len(gen.split()), gen[:80] + ('...' if len(gen) > 80 else ''))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    parser.add_argument('corpus_path', nargs="?", type=str, default='warpeace.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=.0, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
