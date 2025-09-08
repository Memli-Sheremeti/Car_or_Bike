# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: mshereme <mshereme@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/08 13:20:26 by mshereme          #+#    #+#              #
#    Updated: 2025/09/08 14:59:01 by mshereme         ###   ########.fr        #
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
        for i in range(200):  # save first 20 images
            img = dataset_train[i]["image"]
            img.save(f"{dir}/img_{i}.jpg")
    except Exception as e :
        print(f'Error as {e}')
    
def main():
    try:
        image_to_test(dir="data/car", src_data="fashxp/cars-manufacturers")
        image_to_test(dir="data/bike", src_data="Ketansomewhere/bikes")
    except Exception as e:
        print(f'Error as {e}')
    
    dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
    ).dataloaders("data", bs=32)
    dls.show_batch(max_n=6)


if __name__ == "__main__":
    main()
