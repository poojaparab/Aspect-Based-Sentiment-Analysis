# Aspect Based Sentiment Analysis

Welcome to the Aspect Based Sentiment Analysis project! This repository contains an algorithm for performing aspect-based sentiment analysis, and it is hosted on AWS SageMaker for easy deployment and scaling.

## Overview

Aspect Based Sentiment Analysis is the process of identifying and extracting specific aspects or features mentioned in a text, along with the sentiments associated with those aspects. For instance, given a statement like: "Great food at a great price! Love the fish plates as well as the salads. Chain restaurant that doesn't feel like a chain! Love this place!" the algorithm will output sentiment polarities for various aspects and corresponding opinions, such as:

```
{
  "aspect=food, opinion=great": "polarity=0.9",
  "aspect=price, opinion=great": "polarity=0.9",
  "aspect=fish, opinion=love": "polarity=0.9",
  "aspect=love, opinion=place": "polarity=0.9"
}
```

## Implementation

The aspect-based sentiment analysis algorithm follows these key steps:

1. **Coreference Resolution:** Utilizes neuralcoref library to perform coreference resolution, ensuring that pronouns and references are correctly associated with their intended nouns.

2. **Stanford CoreNLP (Stanza):** Employs the Stanza library, previously known as Stanford CoreNLP, for text preprocessing and dependency parsing. This aids in extracting grammatical relationships between words.

3. **Grammar Rules:** Implements five basic grammar rules to extract aspects and opinions related to those aspects from the sentences.

4. **TextBlob for Polarity:** Applies TextBlob, a library for processing textual data, to extract the polarity (sentiment) of both the aspects and opinions. The polarity ranges from 0 (negative) to 1 (positive).

## FastAPI Implementation

For a more detailed implementation of the aspect-based sentiment analysis using FastAPI, you can refer to the [GitHub repository](https://github.com/poojaparab/Aspect-Based-Sentiment-Analysis/tree/main/aspect-opinion-extraction-fastapi).

## Objective

The main objectives of this project are:

- Extracting specific aspects mentioned in text.
- Identifying opinions about those aspects.
- Assigning polarity scores to opinions (sentiment analysis).

## Detailed Implementation

The aspect-based sentiment analysis process includes:

1. Utilizing Coreference Resolution (neuralcoref library) to resolve references.
2. Utilizing Stanza (previously Stanford CoreNLP) for preprocessing and dependency parsing.
3. Applying fundamental grammar rules to extract aspects, opinions, and relationships.
4. Utilizing TextBlob to calculate polarity scores for aspects and opinions.
5. Building a custom Docker container for deployment.
6. Deploying the container on AWS SageMaker.
7. Testing the algorithm's robustness using the Yelp dataset.

## Getting Started

To use this aspect-based sentiment analysis algorithm, follow these steps:

1. Clone or download this repository.
2. Set up the required environment and dependencies as mentioned in the repository documentation.
3. Run the provided code to perform aspect-based sentiment analysis on your text data.

Feel free to explore the provided FastAPI implementation for a web-based interface to interact with the algorithm.

For any questions or issues, please refer to the GitHub repository or reach out to the project contributors.

