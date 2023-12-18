Akinator Game Documentation

This project implements a simplified Akinator style game using a decision tree-based approach. The game guesses characters from a dataset based on user input.


Character Database

Number of characters 57 with total number of questions - 76

Name - Character Name
Race - Race of character (Wizard, Human…)
Forces of Good - Is it a good character (yes, no)
Residency/Origins - Where he lives (Gondor, Rivendell…)
Role in Ring Quest - Did he participate in Quest (yes, no)
Interactions with Key Individuals - Did he interact with main characters (yes, no)
Interaction with Ring - Did he interact with Ring (yes, no)
Famous Battle - His most famous battle (Helm’s Deep, Black Gate…)
Unique Abilities - What kind of abilities he has (Swordmanship, Leadership…)
Royal Lineage - Is he part of royal lineage (yes, no)
Mentor/Guide - Did he have a mentor in his journey (yes, no)
Bravery, Loyalty, Wisdom Traits - Does he have such traits (yes, no)
Involvement in Shaping Fate - Did he play crucial role in the story (yes, no)


Optimization and Enjoyability of the Gaming Experience

●	Usage of KnnImputer - Make intermediate predictions to be able to stop and guess the character at some point, so user don’t have to answer to all questions
●	Prints most likely option first, so user don’t have to type a lot of no
●	User can make some mistakes and still be able to get desired answer depending on importance of question
●	Background image in console can dynamically change, so it more enjoyable to users
●	Users can type answers in different forms: 1, Yes, yes, y or 0, No, no, n.                If input is not understandable game will ask to enter answer again
●	After end of the game, Akinator will ask if user wants to play again


Game Concept

●	The Akinator game aims to guess a character from a provided dataset by asking a series of questions
●	Questions are always asked in order. It utilizes a decision tree model to narrow down the possibilities based on user responses
●	There are 2 types of questions : multiple choice questions and simple yes/no questions
●	Users can only answer yes or no 
●	If this is multiple choice question it prints options one by one until user selects one of them
●	It can make some intermediate predictions and put next nearest possible option in the beginning of the list, so there is no need to do a lot of skips during picking the option from multiple choice question
●	If there is enough confidence, the game will stop asking questions and prints: 
“I think this is a __. Am I right? ”
●	If yes, then it is success. End current game, ask if user wants to play again
●	If no, delete this character from predictions array and continue asking questions
●	When the game reaches the end. Print final prediction and ask if user wants to play again
●	It is acceptable to make couple of mistakes during the game and still be able to get desirable answer, depending on importance of question
●	Background Image in the console is changed based on the current state of the game (3 states)


Decision Tree Design 

●	Grow tree by splitting data based on features
●	Features refers to the questions in dataset
●	Define best feature to split by calculacting information gain and entropy 
●	Information gain = Entropy(parent) - [weighted average] * Entropy(children)
●	Entropy = -Sum ( #x/n * log2 (#x/n) )
●	Predict by traversing tree to the left or to the right, depending on threshold
●	Return value of the leaf node




Classes

Node: Represents a node in the decision tree.
DecisionTree: Builds the decision tree based on provided data.

Methods:
fit(X, y): Trains the decision tree model using input features (X) and target labels (y)
predict(X): Predicts the character based on user responses
grow_tree(): Recursive function to grow the decision tree
best_split(): Identifies the best feature and threshold for splitting the tree
information_gain(): Calculates the information gain for optimal splitting
entropy(): Calculates the entropy

Functions

get_choice(choice): Handles user input validation for yes/no responses.
getIndex(value, column): Retrieves values from the cell in the data dataset based on value and column number.
getIndex(value, column): Retrieves values from the cell in the df dataset based on value and column number.
get_imputed(arr): Filling with KnnImputer the array with gained answers so far, to make intermediate predictions 
get_options(imputed, i): if it is a multiple choice question returns array options with the most possible option in the beginning of the list. If it a simple question with only two options yes/no, just returns unique values from this column

Files Used

The LOTR.xlsx: Excel file containing character information from "The Lord of the Rings"
names.txt: Text file mapping character numbers after encoding to their names
questions.txt: Text file containing questions used by the game
image.py: change background image in the console
