import pandas as pd  


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import re




happysmileys = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^) 
             :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D""".split()
happypattern = "|".join(map(re.escape, happysmileys))

sadsmileys = """:-( :( :o( :[ :c( :< =[ 8( =( :{ :^(""".split()
sadpattern = "|".join(map(re.escape, sadsmileys))

df = pd.read_csv('twitter_wahl (postgres).csv')
df['pos'] = df['Text'].str.contains(happypattern, regex=True)
df['neg'] = df['Text'].str.contains(sadpattern, regex=True)
df['label'] = 'nosmiley'
df['label'][(df['pos']==True) & (df['neg']==False)] = 'pos'
df['label'][(df['pos']==False) & (df['neg']==True)] = 'neg'
df['label'][(df['pos']==True) & (df['neg']==True)] = 'both'


df4tnt_neg = df[(df['label']== 'neg' )]
df4tnt_pos = df[(df['label']== 'pos' )]
df4tnt_neg['Text'] = df['Text'].str.replace(sadpattern, '')
df4tnt_pos['Text'] = df['Text'].str.replace(happypattern,'')
df4c = df[df['label']=='nosmiley']


textlist_pos = list(df4tnt_pos['Text'])
trainingsize = int(len(textlist_pos) * 0.8)
trainingtext = textlist_pos[:trainingsize]
testtext = textlist_pos[trainingsize:]
classifytext = list(df4c['Text'])
labellist = list(df4tnt_pos['label'])
traininglabel = labellist[:trainingsize]
testlabel= labellist[trainingsize:]

vectorizer = CountVectorizer()
vectorizer.fit(trainingtext)
train_features = vectorizer.transform(trainingtext)

classifier = MultinomialNB()
classifier.fit(train_features.toarray(), traininglabel)

test_features = vectorizer.transform(testtext)

print(classifier.score(test_features, testlabel))

classify_features = vectorizer.transform(classifytext)

df4c['prediction'] = classifier.predict(classify_features)


df4c.to_csv('predicted.csv')

