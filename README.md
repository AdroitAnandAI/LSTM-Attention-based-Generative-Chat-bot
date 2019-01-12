# LSTM Attention based Generative Chat bot
Personified Generative Chatbot using RNNs (LSTM) &amp; Attention in TensorFlow

> “In the next few decades, as we continue to create our digital footprints, millennial's will have generated enough data to make “Digital Immortality” feasible” - *MIT Technology Review, October 2018.*

What if there is ‘life after death’ or you can talk to your loved ones, even after they left you? In the movie Transcendence (2014), AI researcher, Evelyn uploaded her hubby’s consciousness into a quantum computer, just before his imminent death. While the post-death digital interactions with her hubby, remains a science fiction, as of today, **it is feasible to create a digital imprint of you, who can talk just like you!**

Digital “you” may look like a text-based chatbot or an audio voice like Siri or a digitally edited video or a 3-D animated character or even an embedded humanoid robot. **But the main challenge is to personify the multiple personalities, that each one of us are! Yeah, we are different when we talk to different people**, right? This is why the **“Augmented Eternity”** platform takes data from multiple sources — Facebook, Twitter, Messaging apps etc.

Your speech consists of your personal voice and words. State of the art Deep Learning techniques, viz. Sequence to Sequence modelling and Attention models are widely used to 

1. Clone personal voice 

2. Replicate talking style and language

Voice cloning efforts such as Samsung Bixby which aims to preserve voice of our loved ones, or Baidu’s ‘Deep Voice’ AI System (audio samples) addresses first half of the problem. In this blog, **we will focus on the latter half, i.e. to make a personified text-based chatbot, as your digital avatar.**

There are two types of conversation models:

1. **Retrieval-Based**: They use a repository of predefined responses and some kind of heuristics to pick an appropriate response.

2. **Generative**: They generate new responses from scratch, based on Machine Translation techniques. They are much more powerful but hard to perfect. They are more intelligent and advanced, but could make grammatical errors as it doesn’t know rules of language.

## Generative Conversational Modeling Architecture ##

Conversation is an exchange of dialogues. Each dialogue is a “sequence” of words, for which response would be another sequence of words. Thus, to generate suitable response to dialogues, we need to do sequence to sequence modelling. 

*Sequence Modelling Example:*

>Human: How are you? <br/>
>You Bot: I am fine

This is a complex problem as machines neither know language or grammar to make up a sentence, nor context or emotion to generate suitable response. **Recurrent Neural Networks is the go-to architecture to solve Seq2Seq problems**, as other text featurization techniques such as **BOW, TF-IDF, Word2Vec etc, are completely discarding the sequence information**. 

When **RNNs are trained with your personal information** such as Messenger chats, FB/ Twitter comments, SMS conversations etc, then the model would begin to behave like you. More the data, more similar it would be.

## Recurrent Neural Networks & LSTMs ##

Humans don’t start their thinking from scratch every second. As you read this essay, you understand each word based on context, i.e. surrounding words and sentences. Your thoughts have persistence, which defines your learning.

As traditional neural networks cannot do this, Recurrent Neural Networks (RNN) are used. **They are networks with loops, to allow information to persist**. RNNs can be thought of as multiple copies of the same network, each passing a message to a successor.

<p align="center">
    <img src="https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot/blob/master/images/1.png">
</p>

The chain-like structure of RNNs are related to sequences, be it sequence of words, audio or images. For this reason, they are used with incredible success for language modeling, translation, speech recognition etc.

But RNNs struggle to connect information, when the gap grows, between relevant information and the place that it’s needed. **Long Short Term Memory networks (“LSTMs”) are a special kind of RNN, capable of learning long-term dependencies**. 

<p align="center">
    <img src="https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot/blob/master/images/2.png">
</p>

In sequence-to-sequence models (many-to-many RNN), when the size of input and output sequence is different, an **encoder-decoder architecture is used**. Decoder starts only after encoder finishes. **Eg:** Machine Translation, Text Summarization, Conversational Modeling, Image Captioning, and more.

<p align="center">
    <img src="https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot/blob/master/images/3.png">
</p>

**Encoder** units helps to **‘understand’ the input sequence** (“Are you free tomorrow?”) and the **decoder decodes the ‘thought vector’** and generate the output sequence (Yes, what’s up?”). Thought vector can be thought of as **neural representational form of input sequence**, which only the decoder can look inside and produce output sequence.

**To further improve the performance of sequence to sequence models**, attention mechanism is used, in above architecture.

## Attention mechanism ##

A human focuses on specific portions of the image to get the whole essence of the picture. This is essentially how attention mechanism works.

The idea is to **let every step of an RNN pick information from some larger collection of information**. To explicate, for image captioning problem, suitable captions are generated by making the model focus on particular parts of the image, not the whole image. 

<p align="center">
    <img src="https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot/blob/master/images/4.png">
</p>

For language translation, we **take input from each time step of the encoder**, but weightage depends on importance of that time step for the decoder to optimally generate next word in the sequence, as shown in the image below:

<p align="center">
    <img src="https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot/blob/master/images/5.gif">
</p>

Armed with the above techniques, we can now build an end-to-end sequence to sequence model to perform personalized conversational modelling.

## Dataset ##

I have found that **training with only real-word chat conversations between 2 humans doesn't produce stable results**. Chat messages usually contains acronyms (like ‘brb’, ‘lol’ etc), shorthand , net slang and typos to confuse neural network training. Hence, I have used a combination of real-world chat messages and human-bot interactions to train.

1. Personal chat conversations from Whatsapp and Telegram (downloaded as HTML files)

2. The Conversational Intelligence Challenge 2 (ConvAI2) conversational dataset conducted under the scope of NIPS (NeurIPS) 2018 Competition (JSON files)

3. Gunthercox Conversational-dialog engine generated responses based on collections of known conversations to train general purpose conversations like greetings, GK, personal profile etc. (YML files)

4. 157 chats and 6300+ messages with a virtual companion (CSV files)

Some cosmetic modifications have been done in the third and fourth datasets to suit the requirements of the blog.

## Data Preparation & Cleaning ##

As the saying goes, **a model is only as good as the data**. Hence, it is very important to filter and clean the data before feeding the model. 

The incoming data are in various formats, namely, JSON, HTML, YML & CSV. As the YML file is plain-ASCII, all other formats are converted to textual YML. To convert your personal data to input format, you can use the parser code in the project, i.e. convert_html2yml.py or convert_json2yml.py. If your personal data is in some other format, you may use a YML conversion tool or write your own code to format the data as in YML files in \data folder.

**a) JSON Parsing**

ConvAI2 dataset downloaded as 3 JSON files are parsed using Python.

**b) HTML Parser**

The downloaded messenger data from Telegram and Whatsapp were saved as HTML and Text files. All the HTML files are parsed using the Python function below. Text file processing is more straightforward using username tokens.

The conversation of **both individuals are marked with different symbols to indicate input & target sequence**. Special characters, web links, date, time, bot-specific markers & emoji’s are filtered. CSV files are processed using Excel.

**c) YML Parser**

After (a)&(b), all the files are converted to YML format. As this specific YML dataset contains just ‘list’ notations, we use a simple file parser to take in the data. The max length of sequence is fixed and the words are counted to filter out rare words. The below function is used to parse the whole data set.

## Data Pre-Processing ##

After filtering out rare words, we will create dictionaries to provide a unique integer for each word. **Forward and Reverse mapping dictionaries for Word2Index and Index2Word conversion** is created. 

Input sequence (words) are converted to indices using Word2Index and are padded to same length, for batch input to encoder. Output from encoder are converted from integer to words using Index2Word mapping. Add a special token in front of all target data to signal the decoder.

<p align="center">
    <img src="https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot/blob/master/images/6.png">
</p>

## Building Sequence2Sequence Model ##

To **train** the model, the padded input and output sequences (indices) from the above step are fed to the S2S architecture below. The embedding layer convert words to indices (already done above) which are **fed to multiple LSTM cells stacked together in hidden layers**.

<p align="center">
    <img src="https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot/blob/master/images/7.png">
</p>

**Decoding** model can be thought of two separate processes, **training and inference**. During training phase, the input is provided as target label, but in inference phase, the output of each time step will be the input for the next time step. Difference in feed strategy is depicted in the diagram below:

<p align="center">
    <img src="https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot/blob/master/images/8.png">
</p>

The **None** in input placeholder means the batch size, and the batch size is unknown since user can set it. Each input would be of size input_seq_length, after padding. RMSProp optimizer is used, as it is found better than Adam.

**During inference phase, the same model can be reused** with a feed-forward mechanism to predict response sequence (set feed_previous=True)

*Loss corresponding to Number of steps are found to be as below:*

<p align="center">
    <img src="https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot/blob/master/images/9.png">
</p>

**Hyper-parameter tuning** of learning rate, batch size and number of steps are done based on the above plot. For some learning rate, the training loss hikes up after initial fall in loss value. So it is important to **find balance between learning rate and number of steps**.

## Human-Bot Interface ##

To interact with bot, a simple interface is built using Tkinter module in Python

Interestingly, the chat-bot is found to give **responses similar in style to the personal data used for training**. Still there are a few grammatical errors, typical of generative models. But as we add more and more training data & tune the hyper-parameters to minimize the loss value, the bot behaviour is found increasingly stable. To make the response more sensible, we can use **Bidirectional RNNs as it can utilize both past and future context to make a better prediction**, though they are more computationally expensive. 

## Closing Thoughts ##

We’re not there quite yet. A CEO’s digital avatar could just be a **“decision support tool”**, but it won’t be capable of running the company. **Public figures could outsource some of their public interaction**. To consult a famous lawyer, celebrity or politician **would become more feasible & affordable to public**, if we can replicate their digital avatar. You may send your avatar to business meetings, on your behalf. However, it is prudent to admit a truly intelligent Turing-Test-ready bot is as far away a dream as the age foreseen by technological singularity optimists.
