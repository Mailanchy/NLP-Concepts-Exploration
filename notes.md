# Get Into NLP

## Term Frequency - Inverse Document Frequency (TF-IDF)
[Understanding TF-IDF (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/)

* **Term frequency (TF)** is the measure of how often a specific word or term appears in a document.
* **Inverse Document Frequency (IDF)** is a statistical measure that quantifies a term's significance in a document collection by reflecting how common or rare it is across all documents.
* **TF-IDF score** is a multiple of both, and the final score gives a numerical representation of a word's relevance in a document.
* It is a statistical method used in natural language processing to determine the importance of a word to a document within a collection of documents.

### TF-IDF Vectorizer
* The TF-IDF vectorizer (in the scikit library) vectorizes the documents in the given corpus.
* The number of dimensions is equivalent to the number of unique words (excluding stop words) in the corpus.

### Cosine Similarity
* Cosine similarity is a metric that measures the cosine of the angle between two non-zero vectors to gauge their similarity.
* A score of 1 indicates the vectors are identical, 0 means they are orthogonal (unrelated), and -1 means they are in opposite directions.
* Using cosine similarity on the TF-IDF vector, we can find documents which are similar to one another.
* We also can find documents relevant to a specific query.

## Example

**Input:**
> documents = [
>    "I love coding in Python.",
>    "Python is a great tool for Data Science",
>    "I love Data Science.",
>    "The sun is shining today.", 
>    "The cat sat on the mat."
> ]

**Output:**
* **QUERY DOCUMENT (Doc 1):** I love coding in Python.
* **Raw Similarity Scores (Doc 1 vs. All Docs):** `[1.    0.216 0.307 0.    0.   ]`

**BEST MATCH:**
* Document: Doc 3
* Sentence: I love Data Science.
* Score: 0.307

### Analysis
* The higher the score (closer to 1.0), the more similar the TF-IDF vectors are.
* This highly depends on the given corpus.
* *Example:* For the query (doc 1) "I love coding in python", the best match is (doc 3) "I love data science".
* Our human mind finds doc 2 ("Python is a great tool for data science") as the best match.

**Findings on Document Length:**
* The length of a document affects the TF-IDF score of each word in that document (especially the TF score).
* The TF-IDF score for `[doc 1, love]` = `[doc 1, python]` = 0.532.
* Both "love" and "python" repeat 2 times in the whole corpus (Love in doc1 and doc3; python in doc1 and doc2).
* But the TF-IDF score for `[doc 2, python]` < `[doc 3, love]` (0.406 < 0.577).
* This change in score is due to the length of doc 2 (which is 5), being greater than the length of doc 3 (which is 3).
* This score difference is the underlying reason why the similarity score for doc1 and doc3 is higher than doc1 and doc2.

---

# ATTENTION IS ALL YOU NEED
*Research paper by Google Brain in 2017 about transformer architecture.*

* **Sequence transduction model:** Converting an input sequence into an output sequence (e.g., sequence-to-sequence modeling for English to Spanish translation).
* **Language modeling:** A technique in Natural Language Processing (NLP) that involves creating a statistical or machine learning model to predict the probability of a sequence of words occurring in a text.
    * *Example:* It knows "The cat sat on the mat" has a much higher probability than "Mat on the sat cat The."

## Traditional Encoder-Decoder Model
* As we know computers play with numbers, they don’t understand words like humans do.
* We are going to explain how an RNN encoder-decoder model works for translation.

**RNN (Recurrent Neural Network):**
* A type of neural architecture that shares the same set of learned parameters (weights and biases) across all time steps of an input sequence.
* It uses a mechanism of recurrence where the hidden state from the previous time step is fed as an additional contextual input to the current time step's computation.
* *(Less accurate for understanding: identical neural networks are recurrently used one after the other. More technically, RNN has a feedback loop for feeding hidden states back to it).*
[Intro to RNN (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/)

* First, all words have a vector embedding of their own with maybe 50, 100, 300, or whatever dimensions.
* There is a matrix called **E (Embedding matrix)** which consists of embeddings of every known word.

### Encoder Part
When we give the model a sentence (e.g., “The cat ate the mouse”):
1. The embedding for the first word (e.g., 'The') `X1` is passed to the first neural network along with a pre-defined hidden state vector `h0`.
    * In the first state there is no previous hidden state, so we use pre-defined `h0` which is typically a zero vector.
2. By processing `X1` and `h0` (like going through an activation function), we get hidden state `h1`.
3. The next state RNN gets fed by the embedding of the 2nd word `X2` and also the previous hidden state which is `h1`, and processing both results in `h2` (a new hidden state containing info about both words).
4. This process repeats until the given input sentence is completely processed and now we have a hidden state `h-final`.

* What’s happening at state t is Xt and h(t-1) are passed to the network, and ht is the resultant hidden state that has info about all the t words of the sentence: ht = f(Xt, h(t-1)).
* The `h-final` is a vector known as the **"Context vector" (C)** and is passed to the Decoder.

### Decoder Part
1. The context vector `C` is passed to the decoder as initial hidden state `s0`.
2. The first input `Y0` to the decoder is a special token `<SOS>` (start-of-sequence) which lets the decoder know it needs to start generating the resultant sequence.
3. The decoder processes both `s0` and `Y0` to create a new hidden state `s1`.
4. `s1` is passed through a final layer after a softmax function to produce an output probability vector (`P1`).
    * The softmax function is used to get the values of P strictly between 0 and 1.
5. **Output probability vector:**
    * The dimension of the vector is equivalent to the size of the vocabulary the model knows.
    * The vocabulary includes a special token `<EOS>` (end-of-sequence).
    * The vector shows the probability of each word in the vocabulary to be the next actual word (sum of all values = 1, values between 0 and 1).
    * *Example:* If vocabulary is `{the, cat, ate, mouse, EOS}`, then P = {p(the), p(cat), p(ate), p(mouse), p(EOS)}, where p(x) is the probability of word 'x' to be next.
6. The word with the highest value is predicted as the next word.
7. The embedding of this word is `Y1`, which is the next input to the network.
8. For every subsequent step, the Decoder feeds its own previous output back as the current input (**Autoregressive Generation**).
9. The embedding of the pre-predicted word and previous hidden state is fed to the network, repeating until the probability for `<EOS>` has the highest value, terminating prediction.
10. The output is the sequence of words predicted by all `P` throughout the process.

### Encoder-Decoder Training
* When the decoder outputs `P` and predicts the next word, it cross-checks with the actual word and an error is calculated.
* This error gets backpropagated all the way to the encoder's first layer, updating weights and biases to generate a better `C` and `P`.
* The technical term for this end-to-end training process is **Backpropagation Through Time (BPTT)**.

---

## Attention Mechanism
*When used with RNN:*
* The encoder part is the same as the traditional model.
* But instead of passing only the final hidden state (Context vector) to the decoder, **all the hidden states** get passed to the decoder for more information.
* The initial state of the decoder is the same (the context vector).
* **`H`:** A set of hidden state vectors of the encoder known as **key vectors**.
* **`s`:** Known as the **query vector**, which is the hidden state of the decoder.

In an RNN with an attention mechanism, there is an extra **attention layer**:
1. In every stage t, the previous hidden state s(t-1) and H are passed to the attention layer.
2. Calculates an alignment score between every key and the query (using dot products or a Feed Forward Network): score(i) = g(s(t-1), hi).
3. The score(i) is passed through a softmax function to generate an attention weight (alpha(i)).
4. A new dynamic context vector c(t) is calculated using a weighted sum of key vectors.
5. The next hidden state is calculated as before using the previous hidden state and Y(t-1).
6. The next word is predicted after concatenating the current hidden state and context vector, passing it to the final layer for `P`, and thereby `Y`.

---

## Introduction to Transformers
* The intro of the paper talks about the drawbacks of pre-existing SOTA technologies like RNN (especially LSTM & GRU) for sequence modeling and transduction problems.

**Drawbacks of Traditional Models:**
* **Sequential limitation:** RNNs process data sequentially (one word/token at a time).
* The calculation for the current step ht must wait for the result of the previous step h(t-1).
* This makes it incapable of parallelization (splitting tasks on modern hardware like GPUs), which takes more time for training.
* **Long-Term Dependency Problem:** The difficulty in connecting distant elements (also known as the vanishing gradient problem).
    * *Example:* "The cat, which was chased by the dog across the muddy field, suddenly ran up the tree."
    * The dependency between 'cat' and 'ran' is important, but the model likely forgets it because the distance is very long.

**The Transformer Architecture:**
* Additional improvements like the attention mechanism allowed modeling of dependencies irrespective of distance, giving models the power to instantly weight the importance of distant words.
* The proposed **Transformer** model is completely based on the attention mechanism and **avoids the use of recurrence**.
* It allows more parallelization by eliminating the sequential constraint, meaning the entire process can run in parallel.
* This results in significantly faster training times and achieves a new state-of-the-art in translation quality.