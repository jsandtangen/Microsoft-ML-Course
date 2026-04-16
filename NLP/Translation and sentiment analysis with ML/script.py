from deep_translator import GoogleTranslator

text = """
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!
"""

translated = GoogleTranslator(source="auto", target="fr").translate(text)

print(translated)