import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import getpass
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

#take raw data
def input_data(fname,limit):
    ini_data = pd.read_csv(fname)
    data = ini_data.head(limit)
    return data

#print few data
def print_data(data,limit):
    print(data.head(limit))

#clean the null values
def clean_null(data):
    data = data.dropna()
    return data

#X training input
def X(data_column):
    X = np.array(data_column)
    return X

#Y the validator (Label)
def Y(data_column):
    Y = np.array(data_column)
    return Y

#tokenize
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words


#preprocess_text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Convert to lowercase
    text = text.lower()

    # Tokenize text into individual words
    words = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Join the processed words back into a single string
    processed_text = " ".join(words)

    return processed_text


#data reading and preprocessing
data = input_data("dataset.csv",1000)
print_data(data,10)
clean_null(data)
data["text_"] = data["text_"].apply(preprocess_text)

#convert x and y into np array
x = X(data["text_"])
y = Y(data["label"])

#feature extraction and transformation
tdif = TfidfVectorizer(tokenizer=tokenizer)
x = tdif.fit_transform(x)

#train & test data spliting
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.5, 
                                                random_state=42)

#model selection
model = RandomForestClassifier()
#training
model.fit(xtrain, ytrain)
#displaying socre
print(model.score(xtest, ytest))

#testing_interface
while True:
    user = input("Enter Product Review --> ")
    if user == "q":
        break
    #preprocess text
    preprodata = preprocess_text(user)
    #features extraction and transformation
    data = tdif.transform([preprodata]).toarray()
    #predict
    output = model.predict(data)
    #display output
    print(output)








"""
#assign meaning to number
data["strength"] = data["strength"].map({0: "Weak", 
                                         1: "Medium",
                                         2: "Strong"})

x = X(data["password"])
y = Y(data["strength"])

#train 
tdif = TfidfVectorizer(tokenizer=word)
x = tdif.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.5, 
                                                random_state=42)
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

#test_interface
while True:
    user = getpass.getpass("Enter Password: ")
    if user == "q":
        break
    data = tdif.transform([user]).toarray()
    output = model.predict(data)
    print(output)

"""

















































































































"""
ini_data = pd.read_csv("data.csv", error_bad_lines=False)
data = ini_data.head(50000)
print(data.head())
data = data.dropna()

data["strength"] = data["strength"].map({0: "Weak", 
                                         1: "Medium",
                                         2: "Strong"})

print(data.sample(5))

def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
  

x = np.array(data["password"])
y = np.array(data["strength"])

tdif = TfidfVectorizer(tokenizer=word)
x = tdif.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.05, 
                                                random_state=42)
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


user = getpass.getpass("Enter Password: ")
data = tdif.transform([user]).toarray()
output = model.predict(data)
print(output)
"""