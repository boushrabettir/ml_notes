# Introduction P-1

1. CNN - In CNN usually used for Image data, we do not need prev/future data when we dont care about the past and the future
2. Then came out RNN for Sequential Data - Sequential data doesn't work well with CNN since the past+future matter (E.g. If you watch the third HP movie, youll be confused since you dont know what happened in the past two movies, so in other words the current movies past and future matters

## Sequence to Sequence Models
Lets say youre in France and you open the menu and you realize that everything is written in French and you do not know French. So you open up Google Translate, and decode everything and then you order.

That process is similar to Seq-To-Seq Models

In other words they:
1. Solve compelx tasks such as language translation, chatbot, text summ., Q/A system, and much more

```

"HOW ARE YOU" --> SEQ-TO-SEQ MODEL --> "IM DOING FINE"

```

** The most commonly well used model for Sequence to Sequence is ENCODER DECODER **

( View photos on Ipad )

### Step 1
```

Input ---> Word Embeddings ---> Encoder ---> Decoder --->  Output

```

#### Inside the Encoder Decoder Model 
1. Encoder Decoder model
```
                                                                   OUTPUT1   OUTPUT2
                                                                      ^        ^
     RNN -->  RNN  -->  RNN ---> Context Vector (Word Embeddings) --> RNN ---> RNN
      ^        ^         ^
    Input1   Input2    Input3
    
```

2. Teacher Enforcing Encoder Decoder Model
```
                                                                   OUTPUT1   OUTPUT2
                                                                     ^        ^
     RNN -->  RNN  -->  RNN ---> Context Vector (Word Embeddings) --> RNN ---> RNN
      ^        ^         ^                                             ^        ^
    Input1   Input2    Input3                                         Y1        Y2
    
```

- Y1 and Y2 are our tested inputs
- We train the model, so the model compares its predicted output with the trained one
- The optimizer we have optimizes each weight of our NN and then predicts its final output
- So were essentially training the model and then it goes through numerous epochs and its accuracy/loss goes +/- until the last epoch (thats like how i understand it)

### Step 2 - TRANSFORMER MODELS
Transformer Models deal with the following:
1. Limited Memory
2. If text length is larger than 10
3. Larger vocab w/less freq. results in gibberish output
4. Model may overfit

What is a Transformer?
E.g. If we have a model that is trained on a specific corpus (large text or documents) this data will be transformed into for instance, a sentiment analysis score


### Step 3 - A further look into the architecture of the model
(Look at photos on Ipad)


...
### What happens within the RNN?

```
1. Getting ready: The RNN gets prepared to do its job.
(Initialization: The RNN initializes its hidden state.)

2. Understanding words: The RNN changes each word into a special code to show what it means.
(Input Embedding: Each element of the input sequence is transformed into a word embedding.)

3. Figuring out context: The RNN looks at one word at a time and tries to understand what it means in relation to the others.
(Time Step Computation: The RNN processes each input embedding and hidden state to compute a new hidden state.)

4. Remembering information: The RNN remembers what it has learned from previous words.
(Hidden State Update: The computed hidden state becomes the input for the next step, allowing the RNN to remember information from previous steps.)

5. Going through the words: The RNN repeats steps 3 and 4 for every word in the sequence.
(Sequence Iteration: The RNN processes each element in the input sequence, updating its hidden state at each step.)

6. Giving an answer: The RNN may give an answer based on its understanding of the words.
(Output: The RNN produces an output based on the processed sequence, depending on the specific task.)
```

### What happens with the RNN feeds itself to the context vector?
```
1. Word to number: Each word is assigned a special number.
(Vocabulary Representation: Each unique word in the input sequence is assigned a numerical representation.)

2. Special codes for words: The words are changed into special codes that capture their meanings.
(Word Embedding Lookup: Words are replaced with pre-trained vectors that represent their meanings.)

3. Capturing word meanings: The special codes represent the meanings of the words.
(Embedding Transformation: Words are transformed into numerical vectors, which capture their semantic meanings.)

4. Putting codes in order: The codes are arranged in a specific order to show the sequence of words.
(Embedding Sequence: The transformed word embeddings form a new sequence, representing the input words in a specific order.)

Word embeddings help the computer understand the meaning of words by representing them as numerical vectors.
These vectors encode semantic information and allow the machine learning model to process
and analyze the input sequence more effectively.
```

### What happens with the context vector when it feeds itself to the Decoder RNN's?
```
1. Getting ready to understand: The Decoder RNN prepares itself to understand and respond based on the input it received.
(Initialization: The Decoder RNN initializes its hidden state based on the final hidden state of the Encoder RNN or a context vector derived from it.)

2. Talking word by word: The Decoder RNN starts talking and generating words one by one, considering what it has said before and what it knows from the input.
(Decoding: At each step, the Decoder RNN takes the previously generated word, its current understanding (hidden state), and the input information to predict the next word in the response.)

3. Using what it knows: The Decoder RNN uses the information it learned from the input to guide what it says next.
(Contextual Generation: The Decoder RNN uses the context vector, which contains information from the input, to influence how it generates each word, making sure it considers the context of the input.)

4. Keep talking until finished: The Decoder RNN keeps generating words until it decides it has said enough or reaches a certain point.
(Sequence Completion: The generation process continues until a specific condition is met, indicating the completion of the response, such as reaching a maximum length or generating a special end token.)

By feeding the input sequence into the Decoder RNN, it processes the information and generates a response word by word,
considering the context from the input. This enables the model to generate meaningful and contextually relevant responses based on the given input.
```

### What happens when the Decoder RNN's feed itself to....the output?
```
Well, it just outputs its final result.
```


**Theres more that goes to this, I am simply stating the overview of it**
