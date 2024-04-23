import os 
import requests

import tensorflow as tf 


base_url = "http://172.17.0.1:5000"

# UPDATE: Config  ----

# george
# path = "/models/damage/v1"

# path = "/models/hurricane_damage/v1"

# path = "/models/alt_L5/info"
# path = "/models/alt_L5/process"

# path = "/models/hurricane/v1"
# path = "/models/alt_lenet5"

# path = "/model/info"
# path = "/model/predict"

# path = "/info"
# path = "/predict"

# path = "/models"

# path = "/model_summary"
# path = "/classify_image"

# path = "/model/info"
# path = "/model/predict"

path = "/models/best"

# -----------


# final URL 
url = f"{base_url}{path}"


# GET ----
def make_get_request():
    rsp = requests.get(url)
    print(f"Status code: {rsp.status_code}")
    # UPDATE:
    print(rsp.content)
    # print(f"JSON: {rsp.json()}")


# POST -----

def get_paths():
    damage_paths = os.listdir("/data/damage")
    damage_paths = [f"/data/damage/{p}" for p in damage_paths]
    no_damage_paths = os.listdir("/data/no_damage")
    no_damage_paths = [f"/data/no_damage/{p}" for p in no_damage_paths]
    return damage_paths, no_damage_paths


def do_custom_processing(path):
#    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
#    img = tf.keras.utils.load_img(
#        path,
#        color_mode='rgb',
#        target_size=(128,128),
#        interpolation='nearest',
#        keep_aspect_ratio=True
#    )
#    rescale = Rescaling(scale=1.0/255)
#    img_res = rescale(img)
#    tens = tf.convert_to_tensor(img_res)
#    inp = tf.expand_dims(tens, 0, name=None).numpy().tolist()
#    return inp 

    from PIL import Image
    import numpy as np 
    # img = Image.open(path)
    # img = np.asarray(img)
    # return img.tolist()
    img = Image.open(path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_list = np.expand_dims(img_array, axis=0).tolist()
    return img_list
    # from matplotlib import image
    # img = image.imread(path)
    # return img.tolist()


def make_post_request(path):
    
    # UPDATE: Choose one of these ----
    
    # A) read the raw image file in 
    # image = open(path, 'rb').read()

    # B) alternatively, do some custom processing
    image = do_custom_processing(path)
    # send JSON data 
    rsp = requests.post(url, json={"image": image})
    # rsp = requests.post(url, json={"path": path})
    # -----------

    # send multipart
    # data = {"image": image}
    # rsp = requests.post(url, files=data)
    # ------

    # print(f"POST status code: {rsp.status_code}")
    try:
        rsp.raise_for_status()
    except Exception as e:
        print(f"Bad status code: {e}; Response: {rsp.content}")
        return None
    
    # UPDATE: this -----
    result = rsp.json()
    return result

def get_prediction(result, label):
    # UPDATE
    probs = result['result'][0]
    if probs[0] < 0.5:
        prediction = "damage"
    else:
        prediction = "no_damage"
    # if probs[0] > probs[1]:
    #     prediction = "damage"
    # else:
    #     prediction = "no_damage"
    if prediction == label:
        return 1
    else:
        return 0
    
    # prediction = result["result"]
    # prediction = result["prediction"]
    # if "No Damage" in prediction: 
    #     if label == "no_damage":
    #         return 1
    #     else:
    #         return 0
    # elif label == "damage":
    #     return 1
    # else: 
    #     return 0

    # if prediction == "contains no damage" and label == "no_damage":
    #     return 1
    # elif prediction == "contains damage" and label == "damage":
    #     return 1
    # else: 
    #     return 0

def do_full_post_test():
    print("Starting full POST test suite...")
    total_correct = 0
    total = 0

    for p in damage_paths:
        total += 1
        result = make_post_request(p)
        if result:
            prediction = get_prediction(result, "damage")
            total_correct = total_correct + prediction 

    for p in no_damage_paths:
        total += 1
        result = make_post_request(p)
        if result:
            prediction = get_prediction(result, "no_damage")
            total_correct = total_correct + prediction 

    accuracy = float(total_correct)/float(total)

    print("Final results:")
    print(f"Total correct: {total_correct}")
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy}")



print("\n\n\n**** STARTING GRADING ****\n")
make_get_request()
damage_paths, no_damage_paths = get_paths()

# quick test of the POST
print(make_post_request(no_damage_paths[1]))
# for i in range(len(no_damage_paths)):
#     print(make_post_request(no_damage_paths[i]))

# print(make_post_request(damage_paths[2]))

do_full_post_test()

