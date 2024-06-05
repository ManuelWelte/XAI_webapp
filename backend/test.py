import torch
import XAI_core
import datasets
import models

def main():
    model = models.get_model("vgg16", cuda = True)
    dataset = datasets.ImageNetSubset(root = "D:\\imagenet")

if __name__ == "__main__":
    main()


