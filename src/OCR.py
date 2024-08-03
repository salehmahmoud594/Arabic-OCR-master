import cv2 as cv
import os
import time
from tqdm import tqdm
from glob import glob
from character_segmentation import segment
from segmentation import extract_words
from train import prepare_char, featurizer
import pickle
import multiprocessing as mp

# * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
import tensorflow as tf
from tensorflow import keras
from utilities import save_image
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import Sequential, model_from_json
from PIL import Image
from PIL import Image, ImageOps
from functools import partial
import traceback


model_name = "all-char-4k-samples-code-model"  #'2L_NN.sav' #'Enhanced_CNN_Compressed_Dataset_50_Run.sav'
# * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)

classes = [
    "س",
    "و",
    "ظ",
    "ط",
    "غ",
    "ف",
    "ا",
    "ض",
    "ث",
    "ذ",
    "ق",
    "ش",
    "ص",
    "ب",
    "ت",
    "لا",
    "ي",
    "ج",
    "ح",
    "خ",
    "ز",
    "ه",
    "د",
    "ك",
    "م",
    "ر",
    "ل",
    "ن",
    "ع",
]


class CharCount:
    countChar = 0

    @staticmethod
    def inc():
        CharCount.countChar += 1

    all_char_imgs = []


def load_model():
    name = model_name
    location = "models"
    with open(os.path.join(location, f"{name}.json"), "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(location, f"{name}.weights.h5"))
    print(f"Loaded model {name} from disk")
    return loaded_model


# def load_model():
#     location = 'models'
#     if os.path.exists(location):
#         model = pickle.load(open(f'models/{model_name}', 'rb'))
#         return model


def run2(obj, save_imgs, model):
    # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
    word, line = obj
    procees_number = mp.current_process().name.split("-")[1]
    print("run2 save_imgs:", f"{save_imgs}", "process:", f"{procees_number}")
    # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
    # model = load_model()
    # For each word in the image
    char_imgs = []
    # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project) fixing
    try:
        char_imgs = segment(line, word)
    except:
        print("empty line")

    txt_word = ""
    # For each character in the word
    # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
    # CharCount.all_char_imgs.append(char_imgs.copy())
    # print('CharCount.all_char_imgs: ', f'{len(CharCount.all_char_imgs)}', '---',f'{line.id()}')
    for char_img in char_imgs:
        try:
            # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
            # img_path = save_image(char_img, 'chars', f'char{CharCount.countChar}')

            # ready_char = prepare_char(char_img)
            img_path = save_image(char_img, "chars", f"char{CharCount.countChar}")
            my_img = preprocess_image(img_path)
            if save_imgs == 0:
                os.remove(img_path)

            CharCount.inc()
        except:
            # breakpoint()
            traceback.print_exc()
            continue
        # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
        prediction = model.predict(my_img, batch_size=1)
        predicted_class = np.argmax(prediction)
        predicted_char = classes[predicted_class]

        print(
            "prediction:",
            f"{prediction.argmax(axis=-1)}",
            " predicted_class:",
            f"{predicted_class}",
            "predicted_char:",
            f"{predicted_char}",
        )
        # feature_vector = featurizer(ready_char)
        # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
        # print('char_img sh:',f'{np.array(ready_char).shape}','vector:',f'{np.array(feature_vector).shape}')
        # predicted_char = model.predict([feature_vector])[0]

        txt_word += predicted_char
    return txt_word


def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((32, 32))  # Resize to target size
    img = ImageOps.invert(img)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 325.0  # Normalize to [0, 1]
    return img_array


# * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project) cut param


def run(image_path, cut, save_imgs):
    print("run save_imgs:", f"{trace}")

    # Read test image
    full_image = cv.imread(image_path)
    predicted_text = ""
    # Start Timer
    before = time.time()
    # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
    model = load_model()
    # -----
    words = extract_words(
        full_image, 1, cut, save_imgs
    )  # [ (word, its line),(word, its line),..  ]
    pool = mp.Pool(mp.cpu_count())

    # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
    # predicted_words = pool.map(run2, words)
    predicted_words = pool.map(partial(run2, save_imgs=save_imgs, model=model), words)
    # -----
    pool.close()
    pool.join()
    # Stop Timer
    after = time.time()

    # append in the total string.
    for word in predicted_words:
        predicted_text += word
        predicted_text += " "

    # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project)
    print("predicted_text len:", f"{len(predicted_text)}")
    exc_time = after - before
    # Create file with the same name of the image
    img_name = image_path.split("\\")[1].split(".")[0]

    with open(f"output/text/{img_name}.txt", "w", encoding="utf8") as fo:
        fo.writelines(predicted_text)

    return (img_name, exc_time)


def main(cut=3, save_imgs=0):
    global trace
    trace = save_imgs
    print("here save_imgs:", f"{save_imgs}")
    # Clear the old data in running_time.txt
    if not os.path.exists("output"):
        os.mkdir("output")
    open("output/running_time.txt", "w").close()

    destination = "output/text"
    if not os.path.exists(destination):
        os.makedirs(destination)

    types = ["png", "jpg", "bmp"]
    images_paths = []
    for t in types:
        images_paths.extend(glob(f"test/*.{t}"))
    before = time.time()

    # pool = mp.Pool(mp.cpu_count())

    # # Method1
    # for image_path in images_paths:
    #     pool.apply_async(run,[image_path])

    # Method2
    # for _ in tqdm(pool.imap_unordered(run, images_paths), total=len(images_paths)):
    #     pass

    running_time = []

    for images_path in tqdm(images_paths, total=len(images_paths)):
        # * Updated by : NTI Post Graduate Diploma 2023/2024  (Arabic Recognition Project) cut param
        running_time.append(run(images_path, cut, save_imgs))

    running_time.sort()
    with open("output/running_time.txt", "w") as r:
        for t in running_time:
            r.writelines(
                f"image#{t[0]}: {t[1]}\n"
            )  # if no need for printing 'image#id'.

    # pool.close()
    # pool.join()
    after = time.time()
    print(f"total time to finish {len(images_paths)} images:")
    print(after - before)


if __name__ == "__main__":
    main()
