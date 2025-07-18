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

(Replace with your actual scores once you have them)

🧪 How to Run
bash
Copy
Edit
# Clone the repository
git clone https://github.com//cnn-lstm-captioning-scratch.git
cd cnn-lstm-captioning-scratch

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Generate captions
python generate_caption.py --image_path sample.jpg
📝 Project Structure
graphql
Copy
Edit
├── data/               # Raw dataset and processed captions
├── models/             # Saved CNN and LSTM model files
├── utils/              # Utility scripts (tokenizer, data loader, etc.)
├── train.py            # Main training script
├── generate_caption.py # Caption generation for new images
├── requirements.txt
└── README.md
🎯 Future Enhancements
🔍 Add attention mechanism (e.g., Bahdanau)

📈 Improve CNN architecture (e.g., more layers, batch norm)

🚀 Integrate a web interface with Flask/Streamlit

🧠 Experiment with Transformer-based decoders


📬 Contact
Whyshnavi Pathmanathan
📧 Email: whyshnavi01@gmail.com
