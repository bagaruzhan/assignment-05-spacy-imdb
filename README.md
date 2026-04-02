# Assignment 05: Word Embeddings, Order, and the Road to RNNs

**Related to**: Week 8 - Word2Vec and Tokenizers, and preparation for Week 9 RNNs  
**Coursera / CS230 alignment**: C5M2  
**Estimated time**: ~1 week

## Goal

Build intuition for word embeddings, sentence representations, and the limits of order-free text models.

By the end of this assignment, you should be able to:

- preprocess text with spaCy or another standard NLP tool,
- use pretrained word vectors in a simple classifier,
- explain why averaging word vectors can work surprisingly well,
- identify cases where this approach fails because it ignores word order and context,
- connect those failures to why RNNs, LSTMs, and GRUs are useful.

## What you will submit

- One notebook (`.ipynb`) or one script (`.py`) that runs end-to-end
- A short write-up section in markdown answering the follow-up questions
- Required tables / plots / examples from each part

## Rules / constraints

- You may use `spaCy`, `torchtext`, `datasets`, `sklearn`, `PyTorch`, `matplotlib`, `seaborn`, and `pandas`
- Use a validation split, not only train/test
- Set and report a random seed
- Keep the main experiment manageable; do not build a huge pipeline
- You may use spaCy vectors (`en_core_web_md`) or another pretrained embedding source such as GloVe or fastText
- If you use a different dataset from IMDB, explain why and get approval first

---

## Part A: (Paper) Embeddings by Hand: When Averaging Works and When It Breaks

This part should be done by hand or in a clearly written typed solution.

### A1. Toy word vectors

Use the following 3-dimensional toy embeddings:

| Token | Vector |
| --- | --- |
| `dog` | `[1, 0, 1]` |
| `bites` | `[0, 2, 0]` |
| `man` | `[1, 0, -1]` |
| `movie` | `[0, 1, 1]` |
| `good` | `[2, 1, 0]` |
| `not` | `[-2, 0, 0]` |

For each sentence, represent it by the **average** of its word vectors:

1. `"dog bites man"`
2. `"man bites dog"`
3. `"movie good"`
4. `"movie not good"`

### A2. Calculations

Show the sentence vector for all 4 sentences.

Then answer:

1. Which pair has the **same average vector** even though the meaning is different?
2. Which pair has a similar topic but different sentiment?
3. What does this tell you about bag-of-words or average-embedding representations?

### A3. Short interpretation

Write 4-6 sentences explaining:

- when averaging embeddings can still be useful,
- what information is lost,
- why this motivates sequence models.

**Deliverable**: handwritten scan/photo or typed document with all intermediate calculations.

---

## Part B: (Code) Tokenization and Vocabulary Exploration

Use **at least 6 short example sentences** from different situations. Include examples from at least 4 of the following categories:

- movie review,
- negation,
- question / command,
- named entities,
- travel or translation phrase,
- social media / chat style text,
- short spoken-command transcript.

Example sentence ideas:

- `"I loved the movie."`
- `"I did not love the movie."`
- `"Book a flight from Almaty to Astana."`
- `"Turn the light on."`
- `"Can you turn the light off?"`
- `"Apple released a new device in 2025."`

### B1. Compare tokenization

Tokenization means splitting text into smaller units called tokens. In the simplest case, this can be done by splitting a sentence on spaces, for example:  
`"I love this movie"` -> `["I", "love", "this", "movie"]`

You may start with simple space-based tokenization to show the idea, and then compare it with spaCy tokenization.

Example code:

```python
text = "I love this movie"

# simplest version
space_tokens = text.split()
print(space_tokens)
# ['I', 'love', 'this', 'movie']
```

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I love this movie.")

spacy_tokens = [token.text for token in doc]
print(spacy_tokens)
# ['I', 'love', 'this', 'movie', '.']
```

For each example sentence:

- show the raw text,
- show spaCy tokens,
- show lowercase tokens,
- show filtered tokens after removing punctuation and stopwords,
- optionally show lemmas.

### B2. Vocabulary observations

Report:

- total number of tokens,
- number of unique tokens,
- 5 most frequent tokens,
- 5 examples of tokens you would consider noisy or unhelpful,
- at least 3 examples where tokenization matters.

### B3. Word-vector exploration

Pick at least **8 words** from at least **3 semantic groups**, for example:

- emotions: `happy`, `sad`, `angry`
- movies: `actor`, `film`, `director`
- travel: `plane`, `train`, `airport`
- translation-related: `language`, `english`, `kazakh`

For each chosen word:

- inspect whether a pretrained vector exists,
- find its nearest neighbors,
- comment on whether the neighbors make sense.

**Deliverable**: printed examples, one small table, and a short paragraph of interpretation.

---

## Part C: (Code) Static Embedding Baseline for Sentiment Classification

Use the IMDB sentiment dataset.

### C1. Load and inspect the dataset

- Show dataset sizes
- Print 2 positive and 2 negative examples
- Report average review length in tokens

### C2. Preprocess the text

At minimum:

- tokenize (for example, by splitting on spaces, or by using spaCy),
- lowercase,
- remove obvious punctuation,
- decide whether to remove stopwords,
- decide whether to lemmatize.

You must explain your preprocessing choices in 3-5 sentences.

Example preprocessing code:

```python
text = "I did not love this movie!"

# simple split-based version
tokens = text.lower().replace("!", "").split()
print(tokens)
# ['i', 'did', 'not', 'love', 'this', 'movie']
```

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I did not love this movie!")

tokens = [
    token.text.lower()
    for token in doc
    if not token.is_punct
]
print(tokens)
# ['i', 'did', 'not', 'love', 'this', 'movie']
```

### C3. Create one vector per review

Represent each review using a **static embedding summary**. Your baseline must use:

- **mean pooling** of pretrained word vectors

You may also compare one additional summary:

- max pooling,
- mean + max concatenation,
- TF-IDF weighted average.

### C4. Train a simple classifier

Train a model that takes the review vector and predicts `positive` or `negative`.

Possible choices:

- logistic regression,
- a small MLP in PyTorch,
- another simple classifier approved by the instructor.

### C5. Evaluation

Report:

- training accuracy,
- validation accuracy,
- test accuracy,
- confusion matrix,
- 5 correctly classified reviews,
- 5 incorrectly classified reviews.

For each error example, include a 1-2 sentence guess about why the model failed.

---

## Part D: (Code + Short Writing) Stress Test the Model: Where Order and Context Matter

This is the most important conceptual part of the assignment.

Create a **contrast set** of at least **12 short examples** where word order, negation, or context changes the meaning.

Your set must include at least:

- 3 negation examples,
- 3 word-order examples,
- 2 ambiguity / word-sense examples,
- 2 examples inspired by translation or bilingual meaning,
- 2 examples of your own choice.

You may use examples such as:

- `"The movie was good."` vs `"The movie was not good."`
- `"Dog bites man."` vs `"Man bites dog."`
- `"I hardly liked it."` vs `"I liked it."`
- `"Book a flight."` vs `"Read a book."`
- `"He went to the bank."` vs `"He sat by the river bank."`
- `"Flights from Astana to Almaty."` vs `"Flights from Almaty to Astana."`

### D1. Run your baseline on the contrast set

For each example:

- compute the sentence vector,
- run the classifier or similarity pipeline,
- record the model output.

### D2. Analyze failures

Identify at least **4 cases** where the representation or model is misleading, unstable, or clearly wrong.

### D3. Write the bridge to RNNs

Write one paragraph answering:

- Why does averaging lose order information?
- Why can static vectors struggle with polysemy?
- What kind of hidden state should a sequence model keep that a bag-of-embeddings model cannot?

**Deliverable**: one table of your contrast set and a short analysis paragraph.

---

## Part E: (Code + Short Writing) Choose One Bridge Mini-Track

Choose **one** of the following mini-tracks. The goal is not to build a full RNN yet, but to prepare for one.

### Option 1. Translation Mini-Track

Use a tiny phrase-level dataset such as English-Kazakh, English-Russian, or English-Turkish phrase pairs.

You may create your own small dataset of **50-150 phrase pairs** if needed.

Tasks:

1. Show 10 example source-target pairs.
2. Tokenize both source and target sides. A simple first version is to split each phrase on spaces.
3. Add special tokens such as `<SOS>`, `<EOS>`, and `<PAD>`.
4. Plot or report the distribution of sequence lengths.
5. Show one padded batch and explain why padding is needed.
6. Explain why averaging source-word vectors is not enough to generate a correct target sequence.

Examples:

- `"good morning" -> "kaiyrly tan"`
- `"thank you" -> "rahmet"`
- `"where is the station?" -> "..."`
- `"book a ticket" -> "..."`

Example code:

```python
pairs = [
    ("good morning", "kaiyrly tan"),
    ("thank you", "rahmet"),
]

src_tokens = pairs[0][0].lower().split()
tgt_tokens = pairs[0][1].lower().split()

src_tokens = ["<SOS>"] + src_tokens + ["<EOS>"]
tgt_tokens = ["<SOS>"] + tgt_tokens + ["<EOS>"]

print(src_tokens)
print(tgt_tokens)
```

### Option 2. Audio Mini-Track

Use a very small spoken-word or command dataset such as:

- Speech Commands (`yes`, `no`, `up`, `down`),
- spoken digits,
- or your own short recordings.

Tasks:

1. Show 5-10 example audio clips or transcripts.
2. Convert each audio clip into a time-based representation such as waveform, spectrogram, or MFCCs.
3. Visualize at least 3 examples.
4. Create a naive baseline by averaging features over time and applying a simple classifier or nearest-neighbor comparison.
5. Explain what temporal information is lost by averaging.
6. Explain why an RNN / GRU / LSTM would be a natural next step.

Examples:

- `"yes"` vs `"no"`
- `"up"` vs `"down"`
- two speakers saying the same word
- the same word spoken at different speeds

**Deliverable**: 1-2 figures plus a short bridge explanation.

---

## Part F: (Written Answers) Follow-Up Questions for Understanding

Answer all general questions and the questions for the mini-track you chose.

### General questions

1. What is the difference between a **token**, a **type**, and a **vocabulary**?
2. Why can pretrained embeddings help when the dataset is not very large?
3. Why is average pooling over word vectors order-invariant?
4. Give one example where static embeddings are useful and one where they are not enough.
5. What is the difference between a word embedding and a sentence embedding?
6. Why can the word `"bank"` cause trouble for static embeddings?
7. If two sentences contain the same words in a different order, what will happen to a plain average embedding? Why is that a problem?
8. What would you want the hidden state of an RNN to remember while reading a sentence?

### Translation mini-track questions

9. Why do sequence-to-sequence models need `<SOS>` and `<EOS>` tokens?
10. Why do source and target sentences usually need padding in batches?
11. Why is translation a sequence-to-sequence problem rather than a simple classification problem?

### Audio mini-track questions

9. In an audio task, what does one time step represent?
10. Why can two clips of the same word have different lengths?
11. Why is speech naturally a sequential signal rather than a fixed unordered set?

---

## Deliverables Summary

Submit a notebook or script containing:

1. **Part A**: hand-worked sentence-vector calculations done on paper or as a typed calculation sheet
2. **Part B**: tokenization and vocabulary exploration done in code
3. **Part C**: IMDB static-embedding sentiment baseline done in code
4. **Part D**: contrast-set failure analysis done with code plus short written analysis
5. **Part E**: one bridge mini-track (translation or audio) done in code plus short written explanation
6. **Part F**: written answers to the follow-up questions

---

## Resources

- spaCy models and vectors: https://spacy.io/usage/models
- IMDB dataset: https://ai.stanford.edu/~amaas/data/sentiment/
- PyTorch text data utilities: https://pytorch.org/text/stable/
- GloVe vectors: https://nlp.stanford.edu/projects/glove/
- Speech Commands dataset: https://huggingface.co/datasets/google/speech_commands
