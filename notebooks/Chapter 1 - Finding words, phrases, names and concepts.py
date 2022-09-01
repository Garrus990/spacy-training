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

# %%
# Import spaCy
import spacy

# %% [markdown]
# # Introduction

# %% [markdown]
# ### The nlp object

# %% [markdown]
# - contains the processing pipeline
# - includes language-specific rules for tokenization etc.

# %%
# Create a blank English nlp object
nlp = spacy.blank("en")

# %% [markdown]
# ### The Doc object

# %% [markdown]
# Created by processing a string of text with the nlp object.

# %%
doc = nlp("Hello world!")

# Iterate over tokens in a Doc
for token in doc:
    print(token.text)

# %% [markdown]
# ### The Token object

# %%
# Index into the Doc to get a single Token
token = doc[1]

# Get the token text via the .text attribute
print(token.text)

# %% [markdown]
# ### The Span object

# %% [markdown]
# A slice from the Doc is a Span object.

# %%
span = doc[1:3]

# Get the span text via the .text attribute
print(span.text)

# %%
print(f"{span=}")

# %% [markdown]
# ### Lexical attributes

# %%
doc = nlp("It costs $5 (five dollars).")

print("Index:   ", [token.i for token in doc])
print("Text:    ", [token.text for token in doc])

print("is_alpha:", [token.is_alpha for token in doc])
print("is_punct:", [token.is_punct for token in doc])
print("like_num:", [token.like_num for token in doc])

# %% [markdown]
# # Trained pipelines

# %%
# run to get trained pipelines for English
# # !python -m spacy download en_core_web_sm

# %% [markdown]
# What is comprised?
#
# - Binary weights
# - Vocabulary
# - Meta information
# - Configuration file

# %%
import spacy

nlp = spacy.load("en_core_web_sm")

# %% [markdown]
# ### Predicting Part-of-speech Tags

# %%
# Process a text
doc = nlp("She ate the pizza.")

# Iterate over the tokens
for token in doc:
    # Print the text and the predicted part-of-speech tag
    print(token.text, token.pos_, spacy.explain(token.pos_))

# %% [markdown]
# ### Predicting Syntactic Dependencies

# %%
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text, spacy.explain(token.dep_))

# %% [markdown]
# ### Predicting Named Entities

# %%
# Process a text
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")

# Iterate over the predicted entities
for ent in doc.ents:
    # Print the entity text and its label
    print(ent.text, ent.label_, spacy.explain(ent.label_))

# %% [markdown]
# ## Matchers

# %%
import spacy

# Import the Matcher
from spacy.matcher import Matcher

# Load a pipeline and create the nlp object
nlp = spacy.load("en_core_web_sm")

# Initialize the matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# Add the pattern to the matcher
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]
matcher.add("IPHONE_PATTERN", [pattern])

# Process some text
doc = nlp("Upcoming iPhone X release date leaked")

# Call the matcher on the doc
matches = matcher(doc)

# %%
# Iterate over the matches
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)

# %% [markdown]
# ### Matching lexical attributes

# %%
pattern = [
    {"IS_DIGIT": True},
    {"LOWER": "fifa"},
    {"LOWER": "world"},
    {"LOWER": "cup"},
    {"IS_PUNCT": True}
]
matcher = Matcher(nlp.vocab)
matcher.add("world_cup_pattern", [pattern])
doc = nlp("2018 FIFA World Cup: France won!")

matches = matcher(doc)
# Iterate over the matches
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)

# %% [markdown]
# ### Matching other token attributes

# %%
pattern = [
    {"LEMMA": "love", "POS": "VERB"},
    {"POS": "NOUN"}
]

matcher = Matcher(nlp.vocab)
matcher.add("loving_something_pattern", [pattern])

doc = nlp("I loved dogs but now I love cats more.")

matches = matcher(doc)
# Iterate over the matches
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)

# %% [markdown]
# ### Using operators and quantifiers

# %% [markdown]
# - {"OP": "!"}: 	Negation: match 0 times
# - {"OP": "?"}: 	Optional: match 0 or 1 times
# - {"OP": "+"}: 	Match 1 or more times
# - {"OP": "*"}: 	Match 0 or more times

# %%
pattern = [
    {"LEMMA": "buy"},
    {"POS": "DET", "OP": "?"},  # optional: match 0 or 1 times
    {"POS": "NOUN"}
]

matcher = Matcher(nlp.vocab)
matcher.add("buying_something_possibly_with_determiner", [pattern])

doc = nlp("I bought a smartphone. Now I'm buying apps.")

matches = matcher(doc)
# Iterate over the matches
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)
