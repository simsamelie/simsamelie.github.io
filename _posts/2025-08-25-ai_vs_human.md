# Natural Language Processing
> Diving into NLP's to create an AI vs Human text classifier! Exploring fine-tuning pretrained language models lead me into training my very own language model from scratch. This project helped me understand the limitations of certain datasets, and when to accept defeat and try a new approach.

## Text Classification
My next venture into Machine Learning practices is that of NLPs! I decided to build an AI classifier that can distinguish between Human and AI-generated text. Due to the rapid increase in complexity and availability of Generative AI through ChatGPT, Google Overview, and almost any popular software (even on GitHub!), I wanted to use the most recent dataset I could find. [This](https://www.kaggle.com/datasets/pratyushpuri/ai-vs-human-content-detection-1000-record-in-2025/data) Kaggle dataset I found quite interesting. Not only does it contain additional variables along with the text, but the text content itself is written in incomplete sentences, making it almost incoherant. The data rather prioritises specific stylistic patterns found in the full text to summarise the 'writing style'. Since NLPs- or at least the one I will be building- learn language by predicting the next word in the text stream, I'm unsure how the model will react, espeically if the language model is trained on full English sentences. Regardless, I'd like to experiment and see if I can make this work!

## Tokenisation and Fine-tuning
Since the data is much vaster than just the text, I first needed to extract the ```text_content``` column, downloading each input into a text file. This turned out to be quite simple, just using Python's inbuilt file editting system. I later used this again to organise my files in 'AI' and 'Human', which I admittedly should have done right off the bat.

    a = len(data)
    for index in range(a):
      text = str(data.values[index])[2:-2]
      with open("/content/data/file" + str(index) + ".txt", "w") as f:
        f.write(text)

Before we even begin thinking about the classifier, we need to finetune our language model. Fastai provides a default language model, 'Wikitext 103', that is trained to predict the next word of a sentence based on Wikipedia articles. Obviously, these sentences are in complete English, whilst ours aren't. We will take this model and run our own data through it, essentially training it again on new 'unseen' data so it can learn the fundamental style of our particular dataset. For example, if we were building a book review sentiment analyser, the model would have to learn 'meaning' of the particular language used in said reviews like 'plot', 'characters', or 'genre'.

To finetune our model, we first need to convert our data into a format that the model will understand. This is called _tokenisation_. Whilst there are different ways to 'tokenise' a text stream, we will first use _word tokenisation_. This process seperates each unique word into its own 'token', creating a dictionary of words in the data set. Then, we can use _numericalisation_ to convert each word into a unique number and the model will use these numbers to 'learn' our language. We can call this process easily with fastai's ```WordTokenizer()``` and ```Numericalize()``` functions. In addition to tokenising words, the tokeniser also includes 'special tokens'. These correspond mostly to grammar such as ```xxmaj``` meaning the next word starts with a capital letter. 

To see what's going on behind the scenes, it's easiest to give an example. Here, we have a sample from one of our text streams, broken up into words.

```['Road','course','indeed','ability','.','Cost','work','close','.','Must',```
```'smile','while','memory']```

After applying the ```Tokenizer()``` function to include our special tokens, we have

```['xxbos','xxmaj','road','course','indeed','ability','.','\n','xxmaj','cost',```
```'work','close','.','xxmaj','must','smile','while','memory']```

The additional special tokens we see here- ```xxbos``` and ```\n```- correspond to the _beginning of stream_ and a paragraph break respectively.

We can now set up our tokenisation process.

    token = WordTokenizer()
    tkn = Tokenizer(token)

Setting up our data to be tokenised:

    files = get_text_files('/content/data')
    txts = L(o.open().read() for o in files)

Here, I tokenised only a subset of the dataset.

    toks200 = txts[:200].map(tkn)

This is done as to not overfit the language model to the dataset- it needs to be able to perform efficiently on unseen words not present in the dictionary. 

    num = Numericalize()
    num.setup(toks200)

Once we call the numericaliser, we are ready to start finetuning our language model. In a similar way to our celebrity classifier, we will compile the data into a ```DataBlock``` and use fastai's ```language_model_learner``` as follows:

    get_text = partial(get_text_files)

    dls_lm = DataBlock(
        blocks=TextBlock.from_folder('/content/data', is_lm=True),
        get_items=get_text, splitter=RandomSplitter(0.1)
    ).dataloaders('/content/data', path='/content/data', bs=128, seq_len=80)

    learn = language_model_learner(
        dls_lm, AWD_LSTM, drop_mult=0.3, 
        metrics=[accuracy, Perplexity()]).to_fp16()

After a long time of training epochs, we have fine-tuned our language model and are ready to move into classification!

<img width="354" height="259" alt="language model finetuning" src="https://github.com/user-attachments/assets/a91cf414-627d-4828-bca9-35e448c3a4cb" />

## Classification

Here, we will follow a very similar process to the Celebrity Classifier built a few weeks ago. First, we will organise the dataset into a DataBlock, with $20$% split off for validation. Then, we will use fastai's ```text_classifier_learner``` as opposed to ```vision_learner``` used for image classification.

    dls_clas = DataBlock(
        blocks=(TextBlock.from_folder('/content/data', vocab=dls_lm.vocab),CategoryBlock),
        get_y = parent_label,
        get_items=partial(get_text_files, folders=['AI', 'Human']),
        splitter=RandomSplitter(valid_pct=0.2)
    ).dataloaders('/content/data', path='/content/data', bs=128, seq_len=72)

This DataBlock takes all the text files from the AI and Human folders, using the ```parent_label``` again as the classification label.

    learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, 
                                metrics=accuracy).to_fp16()

We define ```learn``` so we can begin training epochs! One final step, we must add in our finetuned language model to our ```learn``` function so it does not result to the default Wikitext model. We simply call ```learn = learn.load_encoder('finetuned')```, where ```finetuned``` is the name of our model.

After training over 10 epochs, the accuracy failed to rise above $51$%. This is not a good sign. Our model isn't even predicting better than normal guesswork! It seems my suspicions of the model performing poorly on incoherant text have proven to be true. However, all is not lost! 

## Building a language model from scratch

Fine-tuning a pretrained language model is effective _if_ the model is suited to your data- clearly, for us, it is not. Let's look for an alternative approach.

The idea that immediately struck to me was 'What if we train our own language model from scratch?'. So, that's exactly what we are going to do. The thought process behind this is, if the model is not immediately exposed to grammatical English sentence structure, will it be as confused by the incoherant texts present in my dataset? My initial concern is that since the text is incoherant, there may not be patterns of specific words occuring togther and thus the model will not really have any foot to stand on. Let's experiment.

We will begin similarly as before, concatenating each text stream to train the language model. Using the fastai framework, the entire dataset was too large to run through the tokeniser together so I used ```random.sample``` as a caveat. However, we will try to use raw Pytorch wherever possible- just to showcase the multiple ways of achieving the same result.

First, we organise the data:

    lines = L()
    for i in range(len(data)):
        with open('/content/data/file' + str(i) + '.txt') as f: lines += L(*f.readlines())

This code does the same job as fastai's ```get_text_files```. 

    random_items = random.sample(lines,850)
    text = ' . '.join([l.strip() for l in random_items]) #joining each piece of text into one long stream

We then proceed as before, tokenising and numericalising the data and creating our vocab! Instead of calling ```num.setup``` as we did before to create our numericalised vocab, let's see how we can do it in Pytorch!

    token = WordTokenizer()
    tkn = Tokenizer(token)
    tokens = tkn(text)

    vocab = L(*tokens).unique() #this line simply collects all unique values found in tokens to create our vocab!

We then use the index of each token in the vocab to become its numericalised form! 

    index = {w:i for i,w in enumerate(vocab)}
    nums = L(index[i] for i in tokens)

The most basic form of language model using a neural network is one that operates in _threes_ - that is a model that works to predict the next word based on the previous three words. The model contains three linear layers- like those we saw in the Titanic classification model- with each layer taking one word as an input. In between these layers, we will again make use of the ReLU function.

    class LMModel1(Module):
    def __init__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)  
        self.h_h = nn.Linear(n_hidden, n_hidden)     
        self.h_o = nn.Linear(n_hidden,vocab_sz)
        
    def forward(self, x):
        h = F.relu(self.h_h(self.i_h(x[:,0])))
        h = h + self.i_h(x[:,1])
        h = F.relu(self.h_h(h))
        h = h + self.i_h(x[:,2])
        h = F.relu(self.h_h(h))
        return self.h_o(h)

Writing our model from scratch, we first initialise the _embedding_ or the vector representation of each word (going from the input to hidden within the network), the _hidden_ linear layer (remaining hidden within the network), and the _output_ linear layer (going from hidden in the network to producing our output). In our ```forward()``` function, we introduce a linear layer with each word, using a ReLU in between each layer. I was quite surprised at how similar the language model worked to our previous, from-scratch classification model since it seemed so much more complicated during the fine-tuning process!

To put this model to work, we need to collect sequences of three consecutive words (or their numericalised form to be exact) and put them into tuples with the proceeding word. We can do this easily with Python.

    seqs = L((tensor(nums[i:i+3]), nums[i+3]) for i in range(0, len(nums)-4,3))

A subset of the ouput: ``` [(tensor([0, 1, 2]), 3),(tensor([3, 4, 5]), 6),(tensor([6, 7, 8]), 1),(tensor([ 1,  9, 10]), 11)...```.

We then split the sequences up randomly to form batches for training. We will use a batch size of $64$ by convention.

    cut = int(len(seqs) * 0.8) #splitting the sequences randomly (fastai does this normally but were working more from scratch)
    dls = DataLoaders.from_dsets(seqs[:cut], seqs[cut:], bs=64, shuffle=False)

And, finally, we can train our language model!

    learn = Learner(dls, LMModel1(len(vocab), 64), loss_func=F.cross_entropy,
                metrics=accuracy)
    learn.fit_one_cycle(10, 1e-3)

<img width="552" height="520" alt="image" src="https://github.com/user-attachments/assets/5c11a104-5ab1-402f-bfac-aa6b7f10c3af" />

After training for 10 epochs, it's safe to say I wasn't feeling confident. The accuracy hadn't improved at all from the finetuned model, and wasn't climbing at speeds I expected from the examples. This dataset really seemed to not enjoy being analysed in this way... but I was lacking confidence in the process entirely. Desperate times call for....

## Finalisation

A new dataset! To see if my choice of data really was the problem, I decided to a seek out a new, more standard dataset. 

- did exactly the same thing with a new dataset and it worked so its obviously the problem with the dataset - it isnt well suited to text processing and classification (which is okay!)
- validation and test accuracy acheived by the model

## Conclusion
- despite the first dataset not really going my way, I dont feel like the time was wasted. Using the original datset to understand the model without achieving a good outcome is okay as it lead me down the avenue of training my own language model, which turned out to be similar to building a linear regression model! (In its structure and use of linear lauers and ReLU functions). I am glad to achieve an actual result at the end too and know it's not a problem with my code, it's just that the initial dataset just wasn't well suite to this type of analysis! I'm glad to learn when to throw in the towel.
