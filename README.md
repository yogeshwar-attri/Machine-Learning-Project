# Machine-Learning-Project
The goal for this project is to use machine learning techniques and algorithms to develop a good prediction model for hotel cancellations and find. The data set is named hotel booking demand and can be found in the following link: https://www.kaggle.com/jessemostipak/hotel-booking-demand. The data set includes approximately 120,000 samples with 32 features. Some of the features included are the number of adults, number of children, number of weeknights booked by the guest, if they are a repeated guest, etc. I have taken the reservation status as our labels. The remaining features are transformed into a numerical representation depending on the algorithm run.
## Perceptron (using Bootstrapping for Cross Validation)
>Linear multiclass perceptron was used. A mean_list and eta_list was created to store average accuracy score and value of the hyperparameter eta during each run of the loop. Here eta represents the constant by which the updates are multiplied during a run of Perceptron.
>Bootstrapping is done 5 times for each value of eta. Bootstrapping is done such that 80% of training data is stored in ‘B_train’, and remaining samples not present in ‘B_train’ are stored in cross validation set ‘B_cross’. 
>The perceptron is fit on ‘B_train’ and tested on ‘B_cross’. The resulting accuracy score is stored and plotted vs eta value to obtain the below figure.
>![](image/1.png)
>From the above graph we see that accuracy is highest for a value of eta=0.4. But since I was Bootstrapping, we can expect slightly different values each time we run the code.
>Using value of eta=0.4 we trained the model on entire training data. Then we ran it on the Test set and obtained a prediction accuracy of 75.37%.
## Adaboost
>For Adaboost we used decision stump as our weak learner. The hyperparameter tuned here was #base learners ranging from 1 to 50. We used k fold (k=5) cross validation to check how many base learners we can use to maximize cross validation prediction accuracy. From the below graph we can see the curve flattens out around 20 base learners. So, we use that value of hyperparameter in our final model.
>![](image/2.png)
>Once we train the Adaboost model using #base leaners = 20 , we check the prediction accuracy on test set which yields 81.2% . Then we plot the ROC curve for each individual class label by varying the offset. From the below graph we see that all 3 classes have a curve above the dotted line which indicates the Adaboost algorithm we trained does a relatively better job than random.
>![](image/3.png)
## SVM
>The sklearn algorithm used to run our classification for SVM was the SVC library. We wanted to achieve a multi-class prediction that can be comparable to the other linear classifiers and so we used a linear kernel for our implementation. When using a radial basis function for our kernel, we achieve subpar accuracies and ROC graphs so sticking with a linear kernel was ideal. Our hyper-parameter for this algorithm was the variable C which is the regularization parameter or slack. We used C values: 0.01, 0.1,0.5,1,2.5,5, and 10 to see if there was a general trend occurring. After running the algorithm, we found that the higher the value C was, the longer it took for the algorithm to fit the training data. At the same time, the larger the training data, the longer SVM took to fit as well. In our implementation we attempted to reduce the amount of time the algorithm took to fit while achieve decent accuracy scores so we settled with a sample size of 1500 (1000 train, 500 test). 
>After running the algorithms and doing k fold cross validation for these different C values, we plotted accuracy scores. From the below graph we see that C=5 has the highest cross validation accuracy and hence we use that in the final model.
>![](image/5.png)
### For C=5 we run SVM and get 79% test accuracy. The ROC curve is seen below:
>![](image/6.png)
