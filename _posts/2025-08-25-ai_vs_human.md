# Natural Language Processing

## Text Classification
My next venture into Machine Learning practices is that of NLPs! I decided to build an AI classifier, that can distinguish the difference between Human and AI-generated text. Due to the rapid increase in complexity and availability of Generative AI through ChatGPT, Google Overview, and almost any popular software (even on GitHub!), I wanted to use the most recent dataset I could find. [This](https://www.kaggle.com/datasets/pratyushpuri/ai-vs-human-content-detection-1000-record-in-2025/data) Kaggle dataset was quite interesting. Not only does it contain additional variables to just the text, but the text content itself is written in incomplete sentences, making it almost incoherant. The data rather prioritises specific stylistic patterns found in the full text to summarise the 'writing style'. Since NLPs- or at least the one I will be building- learn language by predicting the next word in the text stream, I'm unsure how the model will react, espeically if the language model is trained on full English sentences. Regardless, I'd like to experiment and see if I can make this work!

## Tokenisation
Since the data is much vaster than just the text, I first needed to extract the ```text_content``` column, downloading each input into a text file. This turned out to be quite simple, just using Python's inbuilt file editting system. I later used this again to organise my files in 'AI' and 'Human', which I admittedly should have done right off the bat.

    a = len(data)
    for index in range(a):
      text = str(data.values[index])[2:-2]
      with open("/content/data/file" + str(index) + ".txt", "w") as f:
        f.write(text)

Before we even begin thinking about the classifier, we need to finetune our language model. Fastai provides a default language model, 'Wikitext 103', that is trained the predict the next word of a sentence based on Wikipedia articles. Obviously, these sentences are in complete English, whilst ours aren't. We will take this model and run our own data through it, essentially training it again on new 'unseen' data so it can learn the fundamental style of our particular dataset. For example, if we were building a book review sentimental analyser, the model would have to learn the type of particular language used in said reviews like 'plot', 'characters', or '__'.



- tokenisation
- fintetuning

## Fine-tuning the Wikitext 103 language model
- essentially jsut the code process, including freezing
- basically it doesn't work lol

## Building a language model from scratch
