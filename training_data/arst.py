import spacy

nlp = spacy.load("en_core_web_sm")

# Sample text from Pride and Prejudice
text = '''
Mr. Bingley was good-looking and gentlemanlike: he had a pleasant
countenance, and easy, unaffected manners. His sisters were fine women,
with an air of decided fashion. His brother-in-law, Mr. Hurst, merely
looked the gentleman; but his friend Mr. Darcy soon drew the attention
of the room by his fine, tall person, handsome features, noble mien, and
the report, which was in general circulation within five minutes after
his entrance, of his having ten thousand a year. The gentlemen
pronounced him to be a fine figure of a man, the ladies declared he was
much handsomer than Mr. Bingley, and he was looked at with great
admiration for about half the evening, till his manners gave a disgust
which turned the tide of his popularity; for he was discovered to be
proud, to be above his company, and above being pleased; and not all his
large estate in Derbyshire could save him from having a most forbidding,
disagreeable countenance, and being unworthy to be compared with his
friend.
'''

# Process text with spaCy
doc = nlp(text)

# Print named entities
for ent in doc.ents:
    print(ent.text, ent.label_)
