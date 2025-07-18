# CNN-LSTM-Image-captioning
A deep learning project that generates captions for images using a custom-built Convolutional Neural Network (CNN) for visual feature extraction and an LSTM network for sequence generation—without relying on any pre-trained models.

📸 Project Overview
This project demonstrates how to build an image captioning system from the ground up, where both the image encoder (CNN) and language decoder (LSTM) are trained from scratch. The model learns to understand visual content and describe it in natural language.

🧱 Model Architecture
CNN (Custom): Designed and trained from scratch to extract image features.

LSTM: Trained to generate sequences (captions) based on CNN-extracted features and previous words.

Embedding Layer: Converts tokens to dense word vectors.

Dense Output Layer: Predicts the next word in the sequence.

🗂 Dataset
Flickr8k / Flickr30k / MS COCO

Each image has multiple human-annotated captions.

Preprocessing includes:

Lowercasing, removing punctuation

Tokenization and padding

Building vocabulary and word-index mappings

🚀 Features
✅ CNN built and trained from scratch

✅ Custom tokenizer and caption preprocessing

✅ Image feature extraction using custom CNN

✅ Caption generation using LSTM

✅ Evaluation using BLEU score

✅ Support for Greedy and Beam Search decoding

🛠 Technologies Used
Python
TensorFlow / Keras
NumPy, Pandas
NLTK / spaCy
Matplotlib
rough-score

📈 Results
Metric	Score
BLEU-1	0.454458

BLEU-2	0.199821

BLEU-3	0.095696

BLEU-4	0.041299

METEOR  0.146983

ROUGH-L 0.227503

CIDEr   0.099939

📬 Contact
Whyshnavi Pathmanathan
📧 Email: whyshnavi01@gmail.com
