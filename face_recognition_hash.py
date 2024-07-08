import os
import numpy as np
import logging
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.layers import Input, Dense, Flatten
from keras.models import Model, load_model
from sklearn.preprocessing import LabelBinarizer
from keras.models import model_from_json



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_face(filename, required_size=(224, 224)):
    """
    Extracts the face from an image using MTCNN and resizes the image.
    Args:
        filename: The path to the image.
        required_size: The desired size of the face image.
    Returns:
        A NumPy array representing the resized face image.
    """
    try:
        pixels = plt.imread(filename)
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        if not results:
            logging.error(f"No face detected in {filename}")
            return None
        x1, y1, width, height = results[0]['box']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    except Exception as e:
        logging.error(f"Error in extract_face: {e}")
        return None

def get_embeddings(filenames):
    """
    Extracts facial features from a set of images using VGGFace.
    Args:
        filenames: A list of image paths.
    Returns:
        A NumPy array representing the extracted facial features.
    """
    try:
        faces = [extract_face(f) for f in filenames]
        faces = [f for f in faces if f is not None]
        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        embeddings = model.predict(samples)
        return embeddings
    except Exception as e:  
        logging.error(f"Error in get_embeddings: {e}")
        return None

def save_embeddings(embeddings, file_path):
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    return np.load(file_path)

def save_model(model, file_path):
    model.save(file_path)

def load_trained_model(file_path):
    return load_model(file_path)

def HashCode(embeddings, labels):
    """
    Generates supervised hash codes for facial features.
    Args:
        embeddings: A NumPy array representing facial features.
        labels: A NumPy array representing class labels.
    Returns:
        A NumPy array representing the hash codes.
    """
    try:
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Number of embeddings does not match number of labels")
        lb = LabelBinarizer()
        lb.fit(labels)
        labels_binary = lb.transform(labels)
        if labels_binary.shape[1] > 64:
            raise ValueError("Too many classes for 64-bit hashing. Reduce the number of classes or increase hash size")
        labels_binary = np.tile(labels_binary, (1, 64 // labels_binary.shape[1]))
        input_layer = Input(shape=(embeddings.shape[1],))
        flatten_layer = Flatten()(input_layer)
        output_layer = Dense(64, activation='sigmoid')(flatten_layer)
        hash_model = Model(inputs=input_layer, outputs=output_layer)
        hash_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        hash_model.fit(embeddings, labels_binary, epochs=10, batch_size=32)
        binary_codes = hash_model.predict(embeddings)
        return binary_codes, hash_model
    except Exception as e:
        logging.error(f"Error in HashCode: {e}")
        return None, None

def cosine_similarity(emb1, emb2):
    return cosine(emb1, emb2)

def find_similar_images(test_image_path, image_paths, image_embeddings, threshold=0.5, max_results=15):
    """
    Searches for images similar to a test image using cosine similarity.
    Args:
        test_embedding: Embedding of the test image.
        image_paths: A list of image paths.
        image_embeddings: Precomputed embeddings of the images.
        threshold: The similarity threshold.
        max_results: The maximum number of results to return.
    Returns:
        A list of tuples (image path, similarity) representing similar images.
    """
    try:
        test_embedding = get_embeddings([test_image_path])[0]
        similar_images = []
        for img_path, embedding in zip(image_paths, image_embeddings):
            similarity = cosine_similarity(test_embedding, embedding)
            if similarity < threshold:
                similar_images.append((img_path, similarity))
        similar_images.sort(key=lambda x: x[1])
        return similar_images[:max_results]
    except Exception as e:
        logging.error(f"Error in find_similar_images: {e}")
        return []

def display_similar_images(similar_images):
    """
    Displays the similar images.
    Args:
        similar_images: A list of tuples (image path, similarity) representing similar images.
    """
    try:
        plt.figure(figsize=(15, 10))
        columns = 5
        for i, (img_path, similarity) in enumerate(similar_images):
            img = plt.imread(img_path)
            plt.subplot(len(similar_images) // columns + 1, columns, i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error in display_similar_images: {e}")

if __name__ == "__main__":
    try:
        dirName = "images/"
        embeddings_file = './data/embeddings.npy'
        model_file = './models/hash_model.h5'
        
        listOfFiles = [os.path.join(dirpath, file) for dirpath, _, filenames in os.walk(dirName) for file in filenames]

        if os.path.exists(embeddings_file):
            logging.info("Loading embeddings from file...")
            all_embeddings = load_embeddings(embeddings_file)
        else:
            logging.info("Extracting embeddings for all images...")
            all_embeddings = get_embeddings(listOfFiles)
            save_embeddings(all_embeddings, embeddings_file)

        labels = [filename.split('/')[-2] for filename in listOfFiles]
        
        if os.path.exists(model_file):
            logging.info("Loading model from file...")
            hash_model = load_trained_model(model_file)
            hash_codes = hash_model.predict(all_embeddings)
        else:
            logging.info("Generating hash codes...")
            hash_codes, hash_model = HashCode(all_embeddings, np.array(labels))
            save_model(hash_model, model_file)
        
        test_image_path = 'images/yassir/1.jpg'
        # test_image_path = 'images/s10/01.jpg'

        logging.info("Finding similar images...")
        similar_images = find_similar_images(test_image_path, listOfFiles, all_embeddings, threshold=0.5, max_results=15)
        logging.info(f"Found {len(similar_images)} similar images:")
        for img_path, similarity in similar_images:
            logging.info(f"{img_path} with similarity {similarity}")
        display_similar_images(similar_images)
    except Exception as e:
        logging.error(f"Error in main: {e}")

