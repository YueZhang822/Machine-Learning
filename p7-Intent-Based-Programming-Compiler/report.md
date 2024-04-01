# Project Report: Intent-Based Programming Compiler

## Overview:

This project has developed a tool that integrates a large language model (LLM) for intent-based programming, specifically in the networking domain. We introduce a "compiler" capable of processing natural language descriptions or a combination of natural language and Python code to generate executable Python code, effectively bridging the gap between conceptual ideas and technical implementation.

## Description:

The "compiler" parses the input and generates appropriate prompts to engage the LLM, specifically Chat-GPT3.5-Turbo, for context and requirements. It then directs the LLM to generate Python code that reflects the user's intent. To enhance domain awareness, we designed prompts such as "you are skilled in networking," which orient the LLM towards the networking domain.
The "compiler" is designed to work in two modes:

1. Compile Mode: The "compiler" processes a file with a mix of natural language instructions and Python code, converting them into an executable Python file. 
2. Interactive Mode: Designed for real-time use, particularly in Jupyter notebooks. This mode allows users to interactively input natural language instructions and instantly generate corresponding Python code. 

## Example:
Input (natural language):

```
$_$: Split features and labels into train and test, out:features_train, features_test, labels_train, labels_test
$_$: Use the training set to train a RandomForestClassifier with 10 estimators. out:rf
$_$: Test the accuracy of rf using the test set and print out the accuracy. 
```

Output (Python code):

```
# Split features and labels into train and test
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# Create a Random Forest Classifier with 10 estimators
rf = RandomForestClassifier(n_estimators=10, random_state=42)

# Train the model using the training set
rf.fit(features_train, labels_train)
# Test the accuracy of the model
predictions = rf.predict(features_test)
accuracy = accuracy_score(labels_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

## Testing and Results:

We evaluated our compiler's effectiveness using a sample file, raw.py. This file was derived from our solutions to assignment 1 on inferring video quality through network data, by replacing some lines of Python code with plain text instructions. We found that the accuracy of the compiled code in predicting video quality from network data was on par with the original human-written code, standing at 97.8%. The results demonstrate the capability of our "compiler" in accurately interpreting instructions and producing high-fidelity Python code.

## Applications:

Our "compiler" offers dual benefits: Firstly, it empowers individuals with limited programming experience, making programming more accessible to those proficient in networking concepts rather than Python. Secondly, it simplifies complex programming tasks through natural language instructions, enhancing efficiency and allowing users to concentrate on more critical issues. While currently tailored to Python, the principles and methodologies employed could be adapted for other popular programming languages.

## Limitations and Future Work:

1. The quality of code generation heavily depends on the clarity and specificity of the natural language instructions provided. Ambiguous, overly complex, or nonsensical instructions may result in lower quality.
2. The performance of our tool is contingent on the capabilities and limitations of the underlying Large Language Model.