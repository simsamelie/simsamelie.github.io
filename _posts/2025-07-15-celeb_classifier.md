# Image Classification
> Creating my first ever machine learning model, a celebrity lookalike classifier. This project lead me on many different paths, through grappling with a Linux environment to depolying my own web-page.

## An Introduction to Deep Learning

As a mathematics student, I find that I am very accustomed to a bottom-up approach to learning: working from the very base of knowledge into more complex ideas. That's why, when I first inquired into machine learning (particularly that of deep learning), I was intrigued by the [fastai](https://course.fast.ai/) course, 'Practical Deep Learning'. This introductory course takes a top-down approach, getting you stuck in and experimenting from the get go! This is definitely out of my comfort zone and so I decided it was the perfect fit for trying something new. 

However, what I thought was going to be a breezy introduction to this new concept, ended up being a self-led crash course in Linux environments, web development and even lead to me creating this blog! Whilst it took me longer than expected to finally launch my own image classifier, I am sure the skills I have learnt here will serve me as I continue forward.

## Initialising my model

Working through the first two chapters of the fastai course, I worked on creating an image classification model. The examples given in the book are of pet breeds and types of bear- things I thought were pretty easily distinguishable, especially for a person. However, one thing that struck me especially is using a deep learning model to distinguish between things that the average person might have trouble with! I wanted to make something accessible and fun, so I decided to create a Celebrity lookalike identifier. 

Unsure with how the AI would respond to the minutiae of facial features, I decided to begin with 3 celebrities who all have similar features but are relatively easy to tell apart: Anne Hathaway, Julia Roberts and Sandra Bullock.

To begin, I used a package called icrawler to download images from Google and create my own dataset. Starting with 20 images of each celebrity meant that the model was quite inaccurate, so I went up to 50. I didn't want to create a massive dataset, especially since I was particularly focused on understanding the process rather than creating a remarkably accurate model.

## Training

As somebody unfamiliar with the PyTorch library, I found the fastai package a great place to start. It is intuative, especially for somebody proficient in Python, and relatively easy to understand!

I created the dataset in a useable format using fastai's inbuilt dataloader, resizing each image and saving 20% of the data for our validation set.

        dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)

Each element was easily labelled since the icrawler package downloaded each celebrities image into their own folder, hence 'parent_label' did the trick.

<img width="716" height="735" alt="Untitled" src="https://github.com/user-attachments/assets/9e0f314b-0610-458e-a7a3-9e79dc988e4d" />

One thing I noticed immediately about the data is that some photos contained images of multiple people! Cleaning the data is definitely necessary here, but first I must train the model.

      learn = vision_learner(dls, resnet18, metrics=error_rate)
      learn.fine_tune(4) #I used 4 epochs here since 5 lead to some overfitting

After training, I took a look at the confusion matrix and plotted the top losses to see where the model was really getting confused.

<img width="467" height="489" alt="Untitled-1" src="https://github.com/user-attachments/assets/f3897d62-8923-431b-91da-637625b85f59" />

<img width="393" height="802" alt="Untitled" src="https://github.com/user-attachments/assets/9a22cf44-b8c3-4225-80b5-c495c01f7bad" />

As I had imagined, the model was pretty inaccurate: 2 of the top 3 losses (inaccurate predictions where the model was the _most_ certian) contained images of multiple people. In these cases, despite it being multiple photos of the _same_ actress, the model still had an issue. Luckily, fastai has a simple widget to help clean the data. I removed images of multiple people from the data set, leading to the following confusion matrix with the error rate dropping from 0.49 to 0.29.

<img width="467" height="489" alt="Untitled" src="https://github.com/user-attachments/assets/c7e3edf0-0769-432f-9e06-cb581392e113" />

Clearly, the model is still getting confused in places but it's certainly a good start.

Now we know that the model can distinguish faces relatively well, let's try it's luck at some _very_ similar faces: popular celebrity lookalikes, Margot Robbie and Emma Mackey. After following the same steps as before, my new model has an error rate of 0.18! 

<img width="583" height="285" alt="image" src="https://github.com/user-attachments/assets/2b29981a-dc8b-4582-a2e8-a67980130832" />

I find it so interesting that the model had an easier time telling the difference between two people that are genuinely considered to look very alike than it did between three people who are not usually confused! This could be to do with the data, perhaps the two actresses being commonly pictured in different colours or poses, but was interesting nonetheless.

## Deploying

I decided to deploy the more interesting model, the initial experiment, in a Hugging Face space. This is certainly the part of the process that I found the most daunting, having never interacted with a Linux environment before. However, in the process of trying and failing many times to activate my website, I learned a lot about how the environment works, particularly using the terminal efficiently.

Since my model was trained in Google Colab, I had to match the package versions accordingly. The most cumbersome was the version of Python not being as advanced as my local computer in which I had to research and create my own virtual environment using conda and poetry (a Python package used to store packages and package versions for ease of use), which was a suprisingly satisfying conclusion to my trouble with deployment. I now feel well versed in the basics of the Linux environment- using git in the terminal to commit my code to the web page.

You can visit my web page here: [https://ameliesims-celeb-classifier.hf.space](https://ameliesims-celeb-classifier.hf.space)

## Outcome

Throughout this first project, I feel I have learned a lot of valuable, transferable skills- specifically with handling new software and understanding the fundamentals of how machine learning operates in Python. In terms of image classification, I'm interested in applying it towards my interest in transport processes, perhaps using it to identify Turing patterns in nature. I'm also considering taking an image segmentation approach for this, building a model that can identify multiple different pattern types in an image and outline where they are found.
