import image
print("\nWelcome To Akinator Game!")
image.changeBackground(0)

import pandas as pd

df = pd.read_excel("The LOTR.xlsx")

# character names and their numbers after encoding
names = {}
with open('names.txt', 'r') as file:
    for line in file:
        line = line.strip().split(':')
        names[int(line[0])] = line[1]

# list of questions
questions = {}
with open('questions.txt', 'r') as file:
    for line in file:
        line = line.strip().split(':')
        questions[int(line[0])] = line[1]

# creating copy of dataframe so we can use it later
data = df.copy()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

columns = df.columns

X = df.drop('Name', axis=1)
y = df['Name']

X = X.values
y = y.values

import numpy as np
from collections import Counter

class Node: # Node class
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self): # only leaves have value
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.root=None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y) # creating the root of tree

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split): # base case, when we reach leaf
            if len(y) == 0:
                return None 
            return Node(value=y[0])

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False) # selecting column indexes to split

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1) # dividing tree to left and right subtree
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs): # return column index with best information gain
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs) # calculating gain using formula
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y): # calculating entropy using formula
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X]) 

    def _traverse_tree(self, x, node): # traversing tree based on input
        if node.is_leaf_node():
            return node.value

        if node.feature is None or node.threshold is None:
            return 0
    
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    

model = DecisionTree()
model.fit(X,y)

from sklearn.impute import KNNImputer

def getIndex(value,column):
    # get value in corresponding dataframe data, by Labeled option and number of column
    row = df[df[df.columns[column]]==value].index[0]
    return data.iloc[row,column]

def _getIndex(value,column):
    # get value in corresponding dataframe df, by option and number of column
    row = data[data[df.columns[column]]==value].index[0]
    return df.iloc[row,column]

def get_choice(choice):
    # Handling different input types
    if choice=="Yes" or choice=="yes" or choice=="y" or choice=="1":
        return "1"
    elif choice=="No" or choice=="no" or choice=="n" or choice=="0":
        return "0"
    else: 
        print("I don't understand. Please type again")
        sys.stdout.flush()
        return get_choice(input())
    
def get_imputed(arr):
    # Filling arr with nans
    temp = np.full(len(columns)-len(arr)-1,np.nan)
    temp = np.concatenate((arr,temp))
    temp = np.array(temp)
    temp = temp.reshape(1, -1)
    
    # Concatenate arr with X and put in the KnnImputer
    z = np.concatenate( (X, temp) )
    imputer = KNNImputer(n_neighbors=1)
    imputed = imputer.fit_transform(z)
    imputed = imputed[len(imputed)-1]

    return imputed

def get_options(imputed,i):
    # Getting index of next most likely option
    possbile_option = getIndex(imputed[i],i+1)
    next_options = data[columns[i+1]].unique()
    # If multiple choice question put most possible option in front
    if len(next_options)>2:
        index = np.where(next_options==possbile_option)[0][0]
        next_options[index], next_options[0] = next_options[0], next_options[index]
        return next_options
    else:
        return data[columns[i]].unique()
    
import sys



while(True):
    arr = []
    predictions = []
    possbile_option = 0
    end = False

    options = data[columns[1]].unique()
    for i in range(1,len(df.columns)):

        image.changeBackground(1)

        choice = ""

        # Handling questions with Yes/No answers
        if len(df[columns[i]].unique()) == 2: 
            print(f"\n{questions[i]}")
            
            sys.stdout.flush()
            choice = get_choice(input())

            arr.append(1 if choice == "1" else 0)

        # Handling questions with multiple choices
        else:
            print(f"\n{questions[i]} {options[0]}?")

            sys.stdout.flush()
            choice = get_choice(input())
            
            j = 1
            while j<len(options):
                if choice=="1":
                    arr.append(_getIndex(options[j-1],i))
                    break

                print(f"{questions[i]} {options[j]}?")
                                    
                sys.stdout.flush()
                choice = get_choice(input())
                
                if j == len(options)-1: j = 0 
                else: j += 1 

        if i >= 12: break

        imputed = get_imputed(arr) # Getting nearest possible arr with user's answers by the KnnImputer [1,0,1,1]
        options = get_options(imputed,i) # Rearranging the options so that the most likely one will be first
        
        predictions.append(model.predict([imputed])[0]) 
        counter = Counter(predictions) # Counting frequency of predictions, stopping if confident enough
        
        if(counter[predictions[len(predictions)-1]] >= 7):
            print(f"\nI think this is a {names[predictions[i-1]]}. Am I right?")

            str = get_choice(input())
            if str=="1":
                image.changeBackground(2)
                print("\nI'm glad I found your character")
                end = True
                break
            else:
                predictions = [x for x in arr if x != predictions[len(predictions)-1]] #Removing character if guess is wrong
                print("\nLet's try again\n")
                continue
        
        str1 = "█"*i
        str2 = "-"*(len(df.columns)-i-1)
        print(f"Progress: |{str1}{str2}|\n")


    if not end: print(f"\nMy final answer is : {names[model.predict([arr])[0]]}\n")
    print("\nDo you want to play again?")
    choice = get_choice(input())
    if(choice=="0"):
        image.changeBackground(0)
        print("\nSee you soon")
        break
    image.changeBackground(0)