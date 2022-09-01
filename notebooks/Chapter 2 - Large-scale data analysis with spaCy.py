# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data structures

# %% [markdown]
# ### Shared vocab and string store

# %% [markdown]
# - `Vocab`: stores data shared across multiple documents
# - To save memory, spaCy encodes all strings to **hash values**
# - Strings are only stored once in the `StringStore` via `nlp.vocab.strings`
# - String store: **lookup table** in both directions

# %%
import spacy
nlp = spacy.blank("en")

# %%
nlp.vocab.strings.add("coffee")
coffee_hash = nlp.vocab.strings["coffee"]
coffee_string = nlp.vocab.strings[coffee_hash]

# %%
coffee_hash

# %%
coffee_string

# %%
doc = nlp("I love coffee")
print("hash value:", nlp.vocab.strings["coffee"])
print("string value:", nlp.vocab.strings[3197928453018144401])

# %% [markdown]
# The `doc` also exposes the vocab and strings.

# %%
doc = nlp("I love coffee")
print("hash value:", doc.vocab.strings["coffee"])

# %% [markdown]
# ### Lexemes: entries in the vocabulary

# %% [markdown]
# A `Lexeme` object is an entry in the vocabulary

# %%
doc = nlp("I love coffee")
lexeme = nlp.vocab["coffee"]

# Print the lexical attributes
print(lexeme.text, lexeme.orth, lexeme.is_alpha)

# %% [markdown]
# ### The Doc object

# %% [markdown]
# Here we're creating a doc from three words. The spaces are a list of boolean values indicating whether the word is followed by a space. Every token includes that information – even the last one!

# %%
# Create an nlp object
import spacy
nlp = spacy.blank("en")

# Import the Doc class
from spacy.tokens import Doc

# The words and spaces to create the doc from
words = ["Hello", "world", "!"]
spaces = [True, False, False]

# Create a doc manually
doc = Doc(nlp.vocab, words=words, spaces=spaces)

# %% [markdown]
# ### The Spam object

# %%
# Import the Doc and Span classes
from spacy.tokens import Doc, Span

# The words and spaces to create the doc from
words = ["Hello", "world", "!"]
spaces = [True, False, False]

# Create a doc manually
doc = Doc(nlp.vocab, words=words, spaces=spaces)

# Create a span manually
span = Span(doc, 0, 2)

# Create a span with a label
span_with_label = Span(doc, 0, 2, label="GREETING")

# Add span to the doc.ents
doc.ents = [span_with_label]

# %% [markdown]
# ### Word vectors and semantic similarity

# %% [markdown]
# To get word vectors we need at least medium vocabulary (>40 MB data)!

# %%
# !python -m spacy download en_core_web_md

# %%
import spacy

# %%
# Load a larger pipeline with vectors
nlp = spacy.load("en_core_web_md")

# Compare two documents
doc1 = nlp("I like fast food")
doc2 = nlp("I like pizza")
print(doc1.similarity(doc2))

# %%
# Compare two tokens
doc = nlp("I like pizza and pasta")
token1 = doc[2]
token2 = doc[4]
print(token1.similarity(token2))

# %%
# Compare a document with a token
doc = nlp("I like pizza")
token = nlp("soap")[0]

print(doc.similarity(token))

# %%
# Compare a span with a document
span = nlp("I like pizza and pasta")[2:5]
doc = nlp("McDonalds sells burgers")

print(span.similarity(doc))

# %% [markdown]
# **How does spaCy predict similarity?**
#
# - Similarity is determined using word vectors
# - Multi-dimensional meaning representations of words
# - Generated using an algorithm like Word2Vec and lots of text
# - Can be added to spaCy's pipelines
# - Default: cosine similarity, but can be adjusted
# - `Doc` and `Span` vectors default to average of token vectors
# - Short phrases are better than long documents with many irrelevant words

# %%
doc = nlp("I have a banana")
# Access the vector via the token.vector attribute
print(doc[3].vector)
print(doc[3].vector.shape)

# %% [markdown]
# **Similarity depends on the application context**
#
# - Useful for many applications: recommendation systems, flagging duplicates etc.
# - There's no objective definition of "similarity"
# - Depends on the context and what application needs to do

# %%
doc1 = nlp("I like cats")
doc2 = nlp("I hate cats")

print(doc1.similarity(doc2))

# %% [markdown]
# High similarity between the two sentences above is expected, but can be undesired if we are looking for similar sentiment between two sentences.

# %% [markdown]
# ### Efficient phrase matching

# %% [markdown]
# - `PhraseMatcher` like regular expressions or keyword search – but with access to the tokens!
# - Takes `Doc` object as patterns
# - More efficient and faster than the `Matcher`
# - Great for matching large word lists

# %%
from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab)

pattern = nlp("Golden Retriever")
matcher.add("DOG", [pattern])
doc = nlp("I have a Golden Retriever")

# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Get the matched span
    span = doc[start:end]
    print("Matched span:", span.text)

# %%
# Create pattern Doc objects and add them to the matcher
# This is the faster version of: [nlp(country) for country in COUNTRIES]
patterns = list(nlp.pipe(COUNTRIES))
