# Learning rate, batch size, num epochs, image size, etc

import os
from dotenv import load_dotenv

load_dotenv()

TRAIN_ANNOTATIONS_PATH = os.getenv('TRAIN_ANNOTATIONS_PATH')
PARSED_TRAIN_ANNOTATIONS_PATH = os.getenv('PARSED_TRAIN_ANNOTATIONS_PATH')
TRAIN_IMAGES_FOLDER_PATH = os.getenv('TRAIN_IMAGES_FOLDER_PATH')

VAL_ANNOTATIONS_PATH = os.getenv('VAL_ANNOTATIONS_PATH')
PARSED_VAL_ANNOTATIONS_PATH = os.getenv('PARSED_VAL_ANNOTATIONS_PATH')
VAL_IMAGES_FOLDER_PATH = os.getenv('VAL_IMAGES_FOLDER_PATH')

TEST_ANNOTATIONS_PATH = os.getenv('TEST_ANNOTATIONS_PATH')
PARSED_TEST_ANNOTATIONS_PATH = os.getenv('PARSED_TEST_ANNOTATIONS_PATH')
TEST_IMAGES_FOLDER_PATH = os.getenv('TEST_IMAGES_FOLDER_PATH')

# Database Configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'face_recognition_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

BATCH_SIZE = 4
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

IOU_POS_THRESH = 0.5
IOU_NEG_THRESH = 0.4

SAVE_MODEL_PATH = 'best_model.pth'
