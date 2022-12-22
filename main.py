import re
from sklearn.feature_extraction.text import CountVectorizer

# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize

def split_words(sentence):
    """
    Function that allows us to isolate just the words 
    and punctuation in our corpus' sentences.

    Args:
        sentence (string): our line to split
    Returns:
        string[]: list of the words and punctuation in the corpus' sentences 
    """
    # sentence = re.findall(r"[a-zA-Z_0-9']+/[A-Z]+|[-;',.:?!`\"]+/[-;',.:?!`\"]", sentence)
    line = re.sub("[][=]","", sentence)
    line = line.split()
    return [word[:word.find("/")] for word in line]

def split_groups(sentence, stop_words=[]):
    """
    Function that isolates just the nominal groups of words 
    and punctuation in our corpus' sentences.

    Args:
        sentence (string): _description_
        stop_words (string[], optional): words we want to exclude from our output. 
                                         Defaults to [].

    Returns:
        string[]: list of nominal groups of words corpus' sentences
    """
    groups = []
    line = re.sub("[][=]","", sentence) #remove "][="
    words = line.split()
    for word in words:
        if word[:word.find("/")] not in stop_words:
            # "*" are in a word when the word is another occurence of "interest" we are
            # not interested in. (pun non intended)
            if "interest" not in word or "*" in word:
                #take just part of the string after the "/"
                groups.append(word[word.find("/")+1:]) 
            else:
                #we keep "interest" to be able to find its index when bagging
                groups.append(word) 
    return groups


def bag_of_words(n, lines):
    """
    Function that returns the 'n' words/groups before and after the word "interest

    Args:
        n (int): our margin
        lines (string[]): our isolated groups/words that we want to recuperate a 'bag' from

    Returns:
        string[]: the shorted 'lines' input with the 'n' surrounding words of 'interest' (excluding 'interest')
    """
    bag = []
    for words in lines:
        for i,word in enumerate(words):
            #only the 'interest' we want has an "_" (followed by it's meaning number)
            if "_" in word: 
                bag.append([*words[i-n:i],*words[i+1:i+n+1]])
                break
    return bag


def vectorize(n, lines, stop_words=False):
    """Fucntion to vectorize the words in our bags

    Args:
        n (int): our margin
        lines (string[]): our isolated groups/words that we want to recuperate a 'bag' from
        stop_words (bool, optional): if we want to include Scikit's integrated stopwords. 
                                     Defaults to False.

    Returns:
        int[]: our vectored words
    """
    bags = bag_of_words(n,lines)
    cv = CountVectorizer()
    if stop_words:
        cv.stop_words = 'english'
    return cv.fit_transform([ " ".join(bag) for bag in bags])
    
    
with open('corpus.txt') as corpus:
    lines = corpus.read().split('$$')[:-1] #Last element is just a "/n" so we remove it
    
#we add these stopwords for the groups since Scikit's vectorized stop_words function on the groups won't work
with open('stopwords.txt') as stop:
    stop_words = stop.read().split()
    
words_only = [split_words(line) for line in lines]
groups_only = [split_groups(line) for line in lines]
groups_no_stop = [split_groups(line,stop_words) for line in lines]

"""
We have 6 meanings to interest:
    Sense 1 =  361 occurrences (15%) - readiness to give attention 
    Sense 2 =   11 occurrences (01%) - quality of causing attention to be given to 
    Sense 3 =   66 occurrences (03%) - activity, etc. that one gives attention to 
    Sense 4 =  178 occurrences (08%) - advantage, advancement or favor 
    Sense 5 =  500 occurrences (21%) - a share in a company or business 
    Sense 6 = 1252 occurrences (53%) - money paid for the use of money
We create the output labels.
"""
labels = [int(line[line.find("_")+1]) for line in lines]

#Checking whata's the longest line. Answer: 177 words
# lengths = [len(w) for w in words_only]
# print(max(lengths))


# assert labels.count(1) == 361
# assert labels.count(2) == 11
# assert labels.count(3) == 66
# assert labels.count(4) == 178
# assert labels.count(5) == 500
# assert labels.count(6) == 1252

    
