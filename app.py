import os
import sys
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
from flask import Flask, request, jsonify
from kafka import KafkaProducer
import json
import subprocess


# Initialize Kafka producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

def extract_face(filename, required_size=(224, 224)):
    """
    Extracts the face from an image using MTCNN and resizes the image.

    Args:
        filename (str): The path to the image.
        required_size (tuple): The desired size of the face image.

    Returns:
        np.ndarray: A NumPy array representing the resized face image.
    """
    try:
        pixels = plt.imread(filename)
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        if not results:
            logging.error(f"No face detected in {filename}")
            return None
        x1, y1, width, height = results[0]['box']
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
        filenames (list): A list of image paths.

    Returns:
        np.ndarray: A NumPy array representing the extracted facial features.
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
    """Saves embeddings to a file."""
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    """Loads embeddings from a file."""
    return np.load(file_path)

def save_model(model, file_path):
    """Saves a model to a file."""
    model.save(file_path)

def load_trained_model(file_path):
    """Loads a trained model from a file."""
    return load_model(file_path)

def HashCode(embeddings, labels):
    """
    Generates supervised hash codes for facial features.

    Args:
        embeddings (np.ndarray): A NumPy array representing facial features.
        labels (np.ndarray): A NumPy array representing class labels.

    Returns:
        tuple: A tuple containing the hash codes and the hash model.
    """
    try:
        lb = LabelBinarizer()
        lb.fit(labels)
        labels_binary = lb.transform(labels)
        labels_binary = np.tile(labels_binary, (1, 64 // labels_binary.shape[1]))
        input_layer = Input(shape=(embeddings.shape[1],))
        flatten_layer = Flatten()(input_layer)
        output_layer = Dense(64, activation='sigmoid')(flatten_layer)  # 64-bit binary code
        hash_model = Model(inputs=input_layer, outputs=output_layer)
        hash_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        hash_model.fit(embeddings, labels_binary, epochs=10, batch_size=32)
        binary_codes = hash_model.predict(embeddings)
        return binary_codes, hash_model
    except Exception as e:
        logging.error(f"Error in HashCode: {e}")
        return None, None

def cosine_similarity(emb1, emb2):
    """Calculates the cosine similarity between two embeddings."""
    return cosine(emb1, emb2)

def find_similar_images(test_image_path, image_paths, image_embeddings, threshold=0.5, max_results=15):
    """
    Searches for images similar to a test image using cosine similarity.

    Args:
        test_image_path (str): Path to the test image.
        image_paths (list): A list of image paths.
        image_embeddings (np.ndarray): Precomputed embeddings of the images.
        threshold (float): The similarity threshold.
        max_results (int): The maximum number of results to return.

    Returns:
        list: A list of tuples (image path, similarity) representing similar images.
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

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handles image uploads and processes them for facial recognition."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = os.path.join('uploads', file.filename)
    file.save(filename)
    
    # Process the image and get embeddings
    embeddings = get_embeddings([filename])
    
    if embeddings is not None:
        # Convert embeddings to list for JSON serialization
        embeddings_list = embeddings.tolist()
        
        # Send to Kafka topic
        producer.send('facial_recognition_topic', {'filename': file.filename, 'embeddings': embeddings_list})
        
        return jsonify({"message": "File uploaded, processed, and sent to Kafka", "file_path": filename}), 200
    else:
        return jsonify({"error": "Failed to process image"}), 500

@app.route('/find_similar', methods=['POST'])
def find_similar():
    """
    Finds images similar to a given image.

    Args:
        image_path (str): The path to the image.
        threshold (float): The similarity threshold.
        max_results (int): The maximum number of results to return.

    Returns:
        JSON response with similar images.
    """
    data = request.json
    image_path = data['image_path']
    threshold = data.get('threshold', 0.5)
    max_results = data.get('max_results', 15)
    
    embeddings_file = './data/embeddings.npy'
    if os.path.exists(embeddings_file):
        logging.info("Loading embeddings from file...")
        all_embeddings = load_embeddings(embeddings_file)
    else:
        return jsonify({"error": "Embeddings file not found"}), 500

    model_file = './models/hash_model.h5'
    if os.path.exists(model_file):
        logging.info("Loading model from file...")
        hash_model = load_trained_model(model_file)
        hash_codes = hash_model.predict(all_embeddings)
    else:
        return jsonify({"error": "Model file not found"}), 500

    image_paths = np.array([os.path.join(dp, f) for dp, dn, fn in os.walk('images') for f in fn])
    
    similar_images = find_similar_images(image_path, image_paths, all_embeddings, threshold, max_results)
    
    # Send results to Kafka
    producer.send('similar_images_topic', {'query_image': image_path, 'similar_images': similar_images})
    
    return jsonify(similar_images)
    
def start_spark_job():
    spark_submit = "C:/spark-3.5.1/bin/spark-submit" 
    job_script = "./spark_sreaming_app.py" 
    try:
        print(f"Running: {spark_submit} {job_script}")
        subprocess.Popen([spark_submit, job_script])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    os.environ['PYSPARK_PYTHON'] = sys.executable
    start_spark_job()
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
