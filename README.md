# SignOff
a browser-based sign language learning app with real-time camera gesture recognition, built level by level (MediaPipe + TensorFlow.js + Vue).

Sign Off: Model Training Pipeline

This is the model training pipeline for Sign Off.

The goal is to build a lightweight sign language classifier that can eventually run directly in the browser. MediaPipe is used to detect the hand and extract its 21 landmarks. These landmarks are then used as the input to a small classification model.

The basic flow is:

Image / Camera
    ↓
MediaPipe Hands
    ↓
21 hand landmarks
    ↓
Normalize landmarks
    ↓
Classifier
    ↓
Predicted sign

The final model will be converted to TensorFlow.js so that predictions can run on the user's device without sending camera footage to a server.

1. Get the dataset

For the initial version, I'm using ASL-HG from Mendeley Data.

It contains 36,000 images across 36 classes, covering A-Z and 0-9. This makes it useful for the first two levels of Sign Off.

Dataset:

ASL-HG on Mendeley Data

DOI:

ASL-HG DOI

There are raw images as well as a processed version with hand crops and an 80/20 train-test split.

For the first experiment, I don't need all 36 classes. I'll start with a smaller set, such as A-E, and add more classes once the pipeline works.

The dataset should look something like:

data/
  asl_hg/
    A/
      img001.jpg
      img002.jpg
      ...
    B/
      ...
    C/
      ...
    D/
      ...
    E/
      ...

Another dataset I may use later is the American Sign Language A-Z dataset with pre-extracted hand landmarks:

ASL A-Z Dataset + Hand Landmarks on Kaggle

2. Install dependencies

For local development:

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

For the first experiments, I can also run the same pipeline in Google Colab instead of setting everything up locally.

3. Extract hand landmarks

Run:

python scripts/extract_landmarks.py \
  --dataset_dir ./data/asl_hg \
  --out landmarks_dataset.csv

The script runs MediaPipe Hands on each image and extracts the 21 hand landmarks.

Each landmark has:

x
y
z

So each image gives us:

21 landmarks × 3 values = 63 features

The resulting CSV will look roughly like:

x1,y1,z1,x2,y2,z2,...,x21,y21,z21,label
0.42,0.51,-0.02,...,0.31,0.72,-0.04,A
0.38,0.49,-0.01,...,0.29,0.70,-0.03,A
...

Images where MediaPipe cannot find a hand will be skipped and logged.

4. Train the classifier

Once the landmark dataset has been created:

python scripts/train_classifier.py \
  --csv landmarks_dataset.csv \
  --out_dir ./model

The first model will be a small neural network:

63 inputs
   ↓
128 neurons
   ↓
64 neurons
   ↓
number of classes

The final layer uses softmax to produce a probability for each sign.

For example:

A    0.02
B    0.04
C    0.91
D    0.01
E    0.02

The training script should also output basic evaluation results such as accuracy and a confusion matrix so I can see which signs the model struggles with.

The trained model will be saved to:

model/
  saved_model/
  labels.json

labels.json maps the model's class numbers back to the actual signs.

5. Test the model

Before putting the model into the website, I want to test it on images it hasn't seen during training.

The initial split can be:

70% training
15% validation
15% testing

The test set should be kept separate from training so the reported accuracy is actually useful.

For example:

Test accuracy: 94.2%

A: 96%
B: 91%
C: 97%
D: 92%
E: 95%

I'll also check the confusion matrix to see if certain hand shapes are being confused with each other.

6. Convert the model to TensorFlow.js

Once the model is working:

chmod +x scripts/convert_to_tfjs.sh

./scripts/convert_to_tfjs.sh \
  ./model/saved_model \
  ./model/tfjs_model

The resulting model can then be copied into the frontend:

frontend/
  public/
    model/
      model.json
      ...
      labels.json

The browser will load the model and run predictions locally.

7. Connect it to the webcam

The final pipeline in the web app will be:

Webcam
   ↓
MediaPipe HandLandmarker
   ↓
21 landmarks
   ↓
Same normalization used during training
   ↓
TensorFlow.js model
   ↓
Prediction

For example, the frontend could receive:

{
  label: "C",
  confidence: 0.96
}

and display:

Detected: C
Confidence: 96%

The important part is that the preprocessing used in the browser must be the same as the preprocessing used during training.

If the landmark normalization changes during training, the frontend version needs to be updated too.

8. Initial goal

I'm keeping the first version small.

Level 1

Static letters:

A
B
C
D
E

Once that works reliably, expand to:

A-Z
Level 2

Numbers:

0-9
Level 3

Simple signs and words.

This is where the model will eventually need to handle movement rather than just a single static hand position.
