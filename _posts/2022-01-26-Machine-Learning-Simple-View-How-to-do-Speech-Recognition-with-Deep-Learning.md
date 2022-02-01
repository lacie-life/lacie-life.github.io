---
title:  Machine Learning is Fun! - How to do Speech Recognition with Deep Learning
author:
  name: Life Zero
  link: https://github.com/lacie-life
date:  2022-01-26 11:11:11 +0700
categories: [Machine Learning]
tags: [tutorial]
render_with_liquid: false
---

# Machine Learning is Fun!: How to do Speech Recognition with Deep Learning

Speech recognition is invading our lives. It’s built into our phones, our game consoles and our smart watches. It’s even automating our homes. For just $50, you can get an Amazon Echo Dot — a magic box that allows you to order pizza, get a weather report or even buy trash bags — just by speaking out loud:

![Fig.1](https://images.viblo.asia/5fb2289d-cf78-474d-910e-2cce87f44f7a.jpeg)

The Echo Dot has been so popular this holiday season that Amazon can’t seem to keep them in stock!

But speech recognition has been around for decades, so why is it just now hitting the mainstream? The reason is that deep learning finally made speech recognition accurate enough to be useful outside of carefully controlled environments.

Andrew Ng has long predicted that as speech recognition goes from 95% accurate to 99% accurate, it will become a primary way that we interact with computers. The idea is that this 4% accuracy gap is the difference between annoyingly unreliable and incredibly useful. Thanks to Deep Learning, we’re finally cresting that peak.

Let’s learn how to do speech recognition with deep learning!

## Machine Learning isn’t always a Black Box

If you know how neural machine translation works, you might guess that we could simply feed sound recordings into a neural network and train it to produce text:

![Fig.2](https://images.viblo.asia/ee7dd847-122f-4a4d-beb1-d70120a84bfd.png)

That’s the holy grail of speech recognition with deep learning, but we aren’t quite there yet (at least at the time that I wrote this — I bet that we will be in a couple of years).

The big problem is that speech varies in speed. One person might say “hello!” very quickly and another person might say “heeeelllllllllllllooooo!” very slowly, producing a much longer sound file with much more data. Both both sound files should be recognized as exactly the same text — “hello!” Automatically aligning audio files of various lengths to a fixed-length piece of text turns out to be pretty hard.

To work around this, we have to use some special tricks and extra precessing in addition to a deep neural network. Let’s see how it works!

## Turning Sounds into Bits

The first step in speech recognition is obvious — we need to feed sound waves into a computer.

In Part 3, we learned how to take an image and treat it as an array of numbers so that we can feed directly into a neural network for image recognition:

![Fig.3](https://miro.medium.com/max/581/1*zY1qFB9aFfZz66YxxoI2aw.gif)

But sound is transmitted as waves. How do we turn sound waves into numbers? Let’s use this sound clip of me saying “Hello”:

![Fig.4](https://images.viblo.asia/665c9da1-ff8d-47f0-89c9-58a03492db5a.png)

Sound waves are one-dimensional. At every moment in time, they have a single value based on the height of the wave. Let’s zoom in on one tiny part of the sound wave and take a look:

![Fig.5](https://images.viblo.asia/f6e4281c-03b8-4f26-a6fd-12e518487adc.png)

To turn this sound wave into numbers, we just record of the height of the wave at equally-spaced points:

![Fig.6](https://miro.medium.com/max/2000/1*dICZCcmEm_EWWx0yA6B3Cw.gif)

This is called sampling. We are taking a reading thousands of times a second and recording a number representing the height of the sound wave at that point in time. That’s basically all an uncompressed .wav audio file is.

“CD Quality” audio is sampled at 44.1khz (44,100 readings per second). But for speech recognition, a sampling rate of 16khz (16,000 samples per second) is enough to cover the frequency range of human speech.

Lets sample our “Hello” sound wave 16,000 times per second. Here’s the first 100 samples:

![Fig.7](https://images.viblo.asia/276438ac-9567-40f2-883b-25da7b2334e0.png)

### A Quick Sidebar on Digital Sampling

You might be thinking that sampling is only creating a rough approximation of the original sound wave because it’s only taking occasional readings. There’s gaps in between our readings so we must be losing data, right?

![Fig.8](https://images.viblo.asia/ebaa10bc-431b-44c1-9ff8-79b2aea08661.png)

But thanks to the Nyquist theorem, we know that we can use math to perfectly reconstruct the original sound wave from the spaced-out samples — as long as we sample at least twice as fast as the highest frequency we want to record.

I mention this only because nearly everyone gets this wrong and assumes that using higher sampling rates always leads to better audio quality. It doesn’t.

## Pre-processing our Sampled Sound Data

We now have an array of numbers with each number representing the sound wave’s amplitude at 1/16,000th of a second intervals.

We could feed these numbers right into a neural network. But trying to recognize speech patterns by processing these samples directly is difficult. Instead, we can make the problem easier by doing some pre-processing on the audio data.

Let’s start by grouping our sampled audio into 20-millisecond-long chunks. Here’s our first 20 milliseconds of audio (i.e., our first 320 samples):

![Fig.9](https://images.viblo.asia/4b0534da-c610-4410-85b8-6313585234b0.png)

Plotting those numbers as a simple line graph gives us a rough approximation of the original sound wave for that 20 millisecond period of time:

![Fig.10](https://images.viblo.asia/0dc3a38e-41d4-411c-9941-05d1e3b19a7b.png)

This recording is only 1/50th of a second long. But even this short recording is a complex mish-mash of different frequencies of sound. There’s some low sounds, some mid-range sounds, and even some high-pitched sounds sprinkled in. But taken all together, these different frequencies mix together to make up the complex sound of human speech.

To make this data easier for a neural network to process, we are going to break apart this complex sound wave into it’s component parts. We’ll break out the low-pitched parts, the next-lowest-pitched-parts, and so on. Then by adding up how much energy is in each of those frequency bands (from low to high), we create a fingerprint of sorts for this audio snippet.

Imagine you had a recording of someone playing a C Major chord on a piano. That sound is the combination of three musical notes— C, E and G — all mixed together into one complex sound. We want to break apart that complex sound into the individual notes to discover that they were C, E and G. This is the exact same idea.

We do this using a mathematic operation called a Fourier transform. It breaks apart the complex sound wave into the simple sound waves that make it up. Once we have those individual sound waves, we add up how much energy is contained in each one.

The end result is a score of how important each frequency range is, from low pitch (i.e. bass notes) to high pitch. Each number below represents how much energy was in each 50hz band of our 20 millisecond audio clip:

![Fig.11](https://images.viblo.asia/c3af2d37-0714-429f-88d6-dc8395b8b698.png)

But this is a lot easier to see when you draw this as a chart:

![Fig.12](https://images.viblo.asia/efaa09a0-3f71-49b6-8f5b-81ab8f6c46e8.png)

If we repeat this process on every 20 millisecond chunk of audio, we end up with a spectrogram (each column from left-to-right is one 20ms chunk):

![Fig.13](https://images.viblo.asia/73624f82-df13-4696-9c5e-ef562df8c28e.png)

A spectrogram is cool because you can actually see musical notes and other pitch patterns in audio data. A neural network can find patterns in this kind of data more easily than raw sound waves. So this is the data representation we’ll actually feed into our neural network.

## Recognizing Characters from Short Sounds

Now that we have our audio in a format that’s easy to process, we will feed it into a deep neural network. The input to the neural network will be 20 millisecond audio chunks. For each little audio slice, it will try to figure out the letter that corresponds the sound currently being spoken.

![Fig.14](https://images.viblo.asia/1db545d0-0ae6-41c7-8936-412e76834739.png)

We’ll use a recurrent neural network — that is, a neural network that has a memory that influences future predictions. That’s because each letter it predicts should affect the likelihood of the next letter it will predict too. For example, if we have said “HEL” so far, it’s very likely we will say “LO” next to finish out the word “Hello”. It’s much less likely that we will say something unpronounceable next like “XYZ”. So having that memory of previous predictions helps the neural network make more accurate predictions going forward.

After we run our entire audio clip through the neural network (one chunk at a time), we’ll end up with a mapping of each audio chunk to the letters most likely spoken during that chunk. Here’s what that mapping looks like for me saying “Hello”:

![Fig.15](https://images.viblo.asia/ac09690f-a6ef-450b-a06c-c0b13f489f7b.png)

Our neural net is predicting that one likely thing I said was “HHHEE_LL_LLLOOO”. But it also thinks that it was possible that I said “HHHUU_LL_LLLOOO” or even “AAAUU_LL_LLLOOO”.

We have some steps we follow to clean up this output. First, we’ll replace any repeated characters a single character:

- HHHEE_LL_LLLOOO becomes HE_L_LO
- HHHUU_LL_LLLOOO becomes HU_L_LO
- AAAUU_LL_LLLOOO becomes AU_L_LO

Then we’ll remove any blanks:

- HE_L_LO becomes HELLO
- HU_L_LO becomes HULLO
- AU_L_LO becomes AULLO

That leaves us with three possible transcriptions — “Hello”, “Hullo” and “Aullo”. If you say them out loud, all of these sound similar to “Hello”. Because it’s predicting one character at a time, the neural network will come up with these very sounded-out transcriptions. For example if you say “He would not go”, it might give one possible transcription as “He wud net go”.

The trick is to combine these pronunciation-based predictions with likelihood scores based on large database of written text (books, news articles, etc). You throw out transcriptions that seem the least likely to be real and keep the transcription that seems the most realistic.

Of our possible transcriptions “Hello”, “Hullo” and “Aullo”, obviously “Hello” will appear more frequently in a database of text (not to mention in our original audio-based training data) and thus is probably correct. So we’ll pick “Hello” as our final transcription instead of the others. Done!

### Wait a second!

You might be thinking “But what if someone says ‘Hullo’? It’s a valid word. Maybe ‘Hello’ is the wrong transcription!”

Of course it is possible that someone actually said “Hullo” instead of “Hello”. But a speech recognition system like this (trained on American English) will basically never produce “Hullo” as the transcription. It’s just such an unlikely thing for a user to say compared to “Hello” that it will always think you are saying “Hello” no matter how much you emphasize the ‘U’ sound.

Try it out! If your phone is set to American English, try to get your phone’s digital assistant to recognize the world “Hullo.” You can’t! It refuses! It will always understand it as “Hello.”

Not recognizing “Hullo” is a reasonable behavior, but sometimes you’ll find annoying cases where your phone just refuses to understand something valid you are saying. That’s why these speech recognition models are always being retrained with more data to fix these edge cases.

## Can I Build My Own Speech Recognition System?

One of the coolest things about machine learning is how simple it sometimes seems. You get a bunch of data, feed it into a machine learning algorithm, and then magically you have a world-class AI system running on your gaming laptop’s video card… Right?

That sort of true in some cases, but not for speech. Recognizing speech is a hard problem. You have to overcome almost limitless challenges: bad quality microphones, background noise, reverb and echo, accent variations, and on and on. All of these issues need to be present in your training data to make sure the neural network can deal with them.

Here’s another example: Did you know that when you speak in a loud room you unconsciously raise the pitch of your voice to be able to talk over the noise? Humans have no problem understanding you either way, but neural networks need to be trained to handle this special case. So you need training data with people yelling over noise!

To build a voice recognition system that performs on the level of Siri, Google Now!, or Alexa, you will need a lot of training data — far more data than you can likely get without hiring hundreds of people to record it for you. And since users have low tolerance for poor quality voice recognition systems, you can’t skimp on this. No one wants a voice recognition system that works 80% of the time.
For a company like Google or Amazon, hundreds of thousands of hours of spoken audio recorded in real-life situations is gold. That’s the single biggest thing that separates their world-class speech recognition system from your hobby system. The whole point of putting Google Now! and Siri on every cell phone for free or selling $50 Alexa units that have no subscription fee is to get you to use them as much as possible. Every single thing you say into one of these systems is recorded forever and used as training data for future versions of speech recognition algorithms. That’s the whole game!

Don’t believe me? If you have an Android phone with Google Now!, click here to listen to actual recordings of yourself saying every dumb thing you’ve ever said into it:

![Fig.16](https://images.viblo.asia/1b078a2e-b301-47e1-a0ec-404314b0c639.png)

So if you are looking for a start-up idea, I wouldn’t recommend trying to build your own speech recognition system to compete with Google. Instead, figure out a way to get people to give you recordings of themselves talking for hours. The data can be your product instead.

