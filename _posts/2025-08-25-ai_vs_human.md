# Natural Language Processing

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

```['Road','course','indeed','ability','.','Cost','work','close','.','Must','smile','while','memory']```

After applying the ```Tokenizer()``` function to include our special tokens, we have

```['xxbos','xxmaj','road','course','indeed','ability','.','\n','xxmaj','cost','work','close','.','xxmaj','must','smile','while','memory']```

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

Fine-tuning a pretrained language model is effective _if_ the model is suited to your data- clearly, for us, it is not. 
