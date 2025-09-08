# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mshereme <mshereme@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/08 13:20:26 by mshereme          #+#    #+#              #
#    Updated: 2025/09/08 14:32:12 by mshereme         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import time, json
from ddgs import DDGS
from fastcore.all import *
from fastai.vision.all import *
from fastdownload import download_url
from datasets import load_dataset
import os

def search_images(keywords, max_images=200, retries=3, delay=2):
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:  # keep same session
                return L(ddgs.images(keywords, max_results=max_images)).itemgot("image")
        except Exception as e:
            print(f"[Attempt {attempt+1}] Error: {e}")
            time.sleep(delay * (attempt+1))  # exponential backoff
    raise RuntimeError(f"Failed to fetch images for '{keywords}' after {retries} retries.")

def image_to_test(dir, src_data):
    try:
        os.makedirs(dir, exist_ok=True)
        dataset = load_dataset(src_data)
        dataset_train = dataset["train"]
        for i in range(5):  # save first 20 images
            img = dataset_train[i]["image"]
            img.save(f"{dir}/img_{i}.jpg")
    except Exception as e :
        print(f'Error as {e}')
    
def main():
    try:
        image_to_test(dir="data/car_images", src_data="fashxp/cars-manufacturers")
        image_to_test(dir="data/bike_images", src_data="RuudVelo/my_awesome_new_bike")
    except Exception as e:
        print(f'Error as {e}')
    # url = search_images("car", max_images=1)
    # download_url(url[0], dest, show_progress=False)
    # im = Image.open(dest)
    # im.to_thumb(256, 256)
    
    


if __name__ == "__main__":
    main()
