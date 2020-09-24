# Machine-Learning-Project
The goal for this project is to use machine learning techniques and algorithms to develop a good prediction model for hotel cancellations and find. The data set is named hotel booking demand and can be found in the following link: https://www.kaggle.com/jessemostipak/hotel-booking-demand. The data set includes approximately 120,000 samples with 32 features. Some of the features included are the number of adults, number of children, number of weeknights booked by the guest, if they are a repeated guest, etc. I have taken the reservation status as our labels. The remaining features are transformed into a numerical representation depending on the algorithm run.
## Importing the necessary libraries:
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sklearn.neighbors as skln
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

## loading data file
