# Aspect Based Sentiment Analysis
Hello!
This is aspect based sentiment analysis algorithm hosted on AWS sagemaker.
## Overview
If You pass statement like this: "Great food at a great price! Love the fish plates as well as the salads. Chain restaurant that doesn't feel like a chain! Love this place!"
Output will be: {"aspect=food, opinion=great": "polarity=0.9", "aspect=price, opinion=great": "polarity=0.9", "aspect=fish, opinion=love": "polarity=0.9", "aspect=love, opinion=place": "polarity=0.9"}

## Implementation
Using coreference resolution, standford core nlp(currently stanza) and applying 5 basic grammer rules I'm extracticg aspect and opinion about the aspect of the sentences. By using textblob on aspect and opinion, I'm fetching polarity of the words which will be in range from 0(negative) to 1(positive).
