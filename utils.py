# importing libraries
import torch
from torchvision import datasets,transforms
from face_recognition import FaceFeaturesExtractor,preprocessing, FaceRecogniser
from PIL import Image
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import os
from PIL import ImageDraw, ImageFont
def Train_model(path_embeddings,model_path='Train_model.pkl',clf=None):
    embeddings_path = path_embeddings+'embeddings.txt'
    labels_path = path_embeddings+'labels.txt'
    idx_to_class_path = path_embeddings+'idx_to_class.pkl'
    embeddings = np.loadtxt(embeddings_path)
    labels = np.loadtxt(labels_path, dtype='int').tolist()
    idx_to_class = joblib.load(idx_to_class_path)
    if clf is None:
        model= LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
        clf = GridSearchCV(
            estimator=model,
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=3
            )
    clf.fit(embeddings, labels)
    best_model = clf.best_estimator_
    print(clf.cv_results_['mean_test_score'])
    print(clf.cv_results_['std_test_score'])    
    features_extractor = FaceFeaturesExtractor()
    joblib.dump(FaceRecogniser(features_extractor, best_model, idx_to_class), model_path)
    
    
def generate_new_embeddings(image_path,path_embeddings):
    features_extractor = FaceFeaturesExtractor()
    dataset = datasets.ImageFolder(image_path)
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])
    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(img_path)
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)
    Embeddings = np.stack(embeddings)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    np.savetxt(path_embeddings+'embeddings.txt', embeddings)
    np.savetxt(path_embeddings+ 'labels.txt', np.array(labels, dtype=np.str).reshape(-1, 1), fmt="%s")
    joblib.dump(idx_to_class, path_embeddings + 'idx_to_class.pkl')
def draw_bb_on_img(faces, img):
    draw = ImageDraw.Draw(img)
    fs = max(20, round(img.size[0] * img.size[1] * 0.000005))
    font = ImageFont.truetype('fonts/font.ttf', fs)
    margin = 5

    for face in faces:
        print(face.top_prediction.confidence)
        if face.top_prediction.confidence > 0.7:
            text = "%s %.2f%%" % (face.top_prediction.label.upper(), face.top_prediction.confidence * 100)
        else:
            text='Desconocido'
        text_size = font.getsize(text)

        # bounding box
        draw.rectangle(
            (
                (int(face.bb.left), int(face.bb.top)),
                (int(face.bb.right), int(face.bb.bottom))
            ),
            outline='green',
            width=2
        )

        # text background
        draw.rectangle(
            (
                (int(face.bb.left - margin), int(face.bb.bottom) + margin),
                (int(face.bb.left + text_size[0] + margin), int(face.bb.bottom) + text_size[1] + 3 * margin)
            ),
            fill='black'
        )

        # text
        draw.text(
            (int(face.bb.left), int(face.bb.bottom) + 2 * margin),
            text,
            font=font
        )
