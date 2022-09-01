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
# # Processing pipelines

# %%
import spacy
from spacy.language import Language
import pprint

nlp = spacy.blank("en")

# %%
print(nlp.pipe_names)

# %%
nlp = spacy.load("en_core_web_sm")

# %%
print(nlp.pipe_names)

# %%
pprint.pprint(nlp.pipeline)

# %% [markdown]
# ### Adding a custom component

# %%
# Create the nlp object
nlp = spacy.load("en_core_web_sm")

# Define a custom component
@Language.component("custom_component")
def custom_component_function(doc):
    # Print the doc's length
    print("Doc length:", len(doc))
    # Return the doc object
    return doc

# Add the component first in the pipeline
nlp.add_pipe("custom_component", first=True)

# Print the pipeline component names
print("Pipeline:", nlp.pipe_names)

# %%
# process text with custom component
doc = nlp("Hello world!")

# %% [markdown]
# ### Setting custom attributes

# %%
nlp = spacy.load("en_core_web_sm")

# %%
# Import global classes
from spacy.tokens import Doc, Token, Span

# Set extensions on the Doc, Token and Span
Doc.set_extension("title", default=None)
Token.set_extension("is_color", default=False)
Span.set_extension("has_color", default=False)

# %% [markdown]
# **Extension attribute types**
#
# 1. Attribute extensions
# 2. Property extensions
# 3. Method extensions

# %% [markdown]
# #### Attribute extensions

# %% [markdown]
# Attribute extensions set a default value that can be overwritten.

# %%
from spacy.tokens import Token

# Set extension on the Token with default value
Token.set_extension("is_color", default=False, force=True)

doc = nlp("The sky is blue.")

# Overwrite extension attribute value
doc[3]._.is_color = True

# %% [markdown]
# #### Property extensions

# %% [markdown]
# Property extensions work like properties in Python: they can define a getter function and an optional setter.
#
# The getter function is only called when you retrieve the attribute. This lets you compute the value dynamically, and even take other custom attributes into account.

# %%
from spacy.tokens import Token

# Define getter function
def get_is_color(token):
    colors = ["red", "yellow", "blue"]
    return token.text in colors

# Set extension on the Token with getter
Token.set_extension("is_color", getter=get_is_color, force=True)

doc = nlp("The sky is blue.")
print(doc[3]._.is_color, "-", doc[3].text)

# %%
### SPAN EXTENSION
from spacy.tokens import Span

# Define getter function
def get_has_color(span):
    colors = ["red", "yellow", "blue"]
    return any(token.text in colors for token in span)

# Set extension on the Span with getter
Span.set_extension("has_color", getter=get_has_color, force=True)

doc = nlp("The sky is blue.")
print(doc[1:4]._.has_color, "-", doc[1:4].text)
print(doc[0:2]._.has_color, "-", doc[0:2].text)

# %% [markdown]
# #### Method extensions

# %% [markdown]
# Method extensions make the extension attribute a callable method.
#
# You can then pass one or more arguments to it, and compute attribute values dynamically – for example, based on a certain argument or setting.

# %%
from spacy.tokens import Doc

# Define method with arguments
def has_token(doc, token_text):
    in_doc = token_text in [token.text for token in doc]
    return in_doc

# Set extension on the Doc with method
Doc.set_extension("has_token", method=has_token)

doc = nlp("The sky is blue.")
print(doc._.has_token("blue"), "- blue")
print(doc._.has_token("cloud"), "- cloud")

# %% [markdown]
# ## Scaling and performance

# %% [markdown]
# If you need to process a lot of texts and create a lot of `Doc` objects in a row, the `nlp.pipe` method can speed this up significantly.
#
# It processes the texts as a stream and yields `Doc` objects.
#
# It is much faster than just calling nlp on each text, because it batches up the texts.
#
# `nlp.pipe` is a generator that yields `Doc` objects, so in order to get a list of docs, remember to call the list method around it.

# %% [markdown]
# **BAD:**

# %%
docs = [nlp(text) for text in LOTS_OF_TEXTS]

# %% [markdown]
# **GOOD:**

# %%
docs = list(nlp.pipe(LOTS_OF_TEXTS))

# %% [markdown]
# ### Passing in context

# %% [markdown]
# 1. Setting `as_tuples=True` on `nlp.pipe` lets you pass in `(text, context)` tuples
# 2. Yields `(doc, context)` tuples
# 3. Useful for associating metadata with the doc

# %%
nlp = spacy.blank("en")

data = [
    ("This is a text", {"id": 1, "page_number": 15}),
    ("And another text", {"id": 2, "page_number": 16}),
]

for doc, context in nlp.pipe(data, as_tuples=True):
    print(doc.text, context["page_number"])

# %% [markdown]
# Context can also be internalized as attributes of Docs.

# %%
from spacy.tokens import Doc

Doc.set_extension("id", default=None)
Doc.set_extension("page_number", default=None)

data = [
    ("This is a text", {"id": 1, "page_number": 15}),
    ("And another text", {"id": 2, "page_number": 16}),
]

for doc, context in nlp.pipe(data, as_tuples=True):
    doc._.id = context["id"]
    doc._.page_number = context["page_number"]

# %% [markdown]
# ### Using only the tokenizer

# %% [markdown]
# If you only need a tokenized `Doc` object (but not other attributes), you can use the `nlp.make_doc` method instead, which takes a text and returns a doc.

# %%
doc = nlp.make_doc("Hello world!")

# %% [markdown]
# ### Disabling pipeline components

# %%
nlp = spacy.load("en_core_web_sm")

text = "This is a text"

# Disable tagger and parser
with nlp.select_pipes(disable=["tagger", "parser"]):
    # Process the text and print the entities
    doc = nlp(text)
    print(doc.ents)

# %%
import json
import spacy

nlp = spacy.load("en_core_web_sm")

with open("exercises/en/tweets.json", encoding="utf8") as f:
    TEXTS = json.loads(f.read())

# Process the texts and print the adjectives
processed_texts = list(nlp.pipe(TEXTS))
for doc in processed_texts:
  print([token.text for token in doc if token.pos_ == "ADJ"])

# %%
import json
import spacy

nlp = spacy.load("en_core_web_sm")

with open("exercises/en/tweets.json", encoding="utf8") as f:
    TEXTS = json.loads(f.read())

# Process the texts and print the entities
docs = nlp.pipe(TEXTS)
entities = [doc.ents for doc in docs]
print(*entities)

# %%
import spacy

nlp = spacy.blank("en")

people = ["David Bowie", "Angela Merkel", "Lady Gaga"]

# Create a list of patterns for the PhraseMatcher
patterns = nlp.pipe(people)
