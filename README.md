# Emotion Steering Evaluation using transformers

## Overview

This branch introduces a sentiment evaluation framework for our emotion steering project. It assesses how effectively our model can be steered towards or away from specific emotional states.

## Key Features

- Multi-class sentiment analysis using OpenAI's GPT models
- Support for nuanced emotions (e.g., joy, sadness, anger, ambivalence, bittersweet)
    - if you wish to, you can delete ambivalence and bittersweet from the emotion mapping AND the prompt; this will display the emotions that the complex emotion is a composition of (Olson, Sprenkle, & Russell, 1979).
- Multiple iterations per input (n=5) to account for model variance
- Averaging of log probabilities for more robust results

## Quick Start

1. Clone the repo and install dependencies
2. Set up your OpenAI API key in a `.env` file or using ```export OPENAI_API_KEY={key}``` in CLI.
4. Modify `sample_texts` in the script
5. Run `python emotion_steering_eval.py`
6. Check console output and `sentiment_analysis_results.json` for results

## Interpreting Results

- "Most frequent response" indicates dominant sentiment
- Average log probabilities show model's confidence in each emotion category
- Higher log probabilities (closer to 0) indicate stronger confidence

Be wary though, first-token logprobs do not always match the final output of the model (Wang et al. 2024)
