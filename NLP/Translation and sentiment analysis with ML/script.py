from deep_translator import GoogleTranslator
from textblob import TextBlob

text = """
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!
"""

translated = GoogleTranslator(source="auto", target="fr").translate(text)

print(translated)

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

# Translate to French
translated1 = GoogleTranslator(source="auto", target="fr").translate(quote1)
translated2 = GoogleTranslator(source="auto", target="fr").translate(quote2)

# Sentiment (best done in English, but possible after translation too)
sentiment1 = TextBlob(translated1).sentiment
sentiment2 = TextBlob(translated2).sentiment

print(translated1)
print("Sentiment:", sentiment1)

print()

print(translated2)
print("Sentiment:", sentiment2)