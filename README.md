In this problem, you are asked to perform the wrapper-type feature selection using the Naïve Bayes classifier for cancer dataset (Breast Cancer Wisconsin (Original)) Dataset. You can download the dataset from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29). In this problem, we want to select 3 attributes out of 9. To begin one experiment, randomly draw 60 % of the instances from each class for training, and 20% from each class for finding the best 3 attributes. Once the feature selection is complete, use the rest 20% for testing to obtain the accuracy. Repeat the selection 10 times to get the average accuracy. Compare the obtained accuracy with the same type of model trained with the full set of features. Note that we just want to keep 9 attributes out of 10. (hint: one attribute is useless. Which one is it?) In addition, this dataset has missing attributes. Explain how you handle missing attributes. You can use sklearn to simplify the programming burden.

![Screenshot 2024-03-24 at 7 59 59 PM](https://github.com/bcchang83/Naive-Bayes-Classifier-with-Breast-Cancer-Dataset/assets/54743478/09871b1a-9299-452e-b587-9be56e86ec8e)

We need to drop out the attribute Sample_code_number, because it's a series of IDs.

In the first experiment, I tried to drop out some data with missing attribute, and compared the result of features selection with the result full features.

![Screenshot 2024-03-24 at 8 04 41 PM](https://github.com/bcchang83/Naive-Bayes-Classifier-with-Breast-Cancer-Dataset/assets/54743478/e6cc4d85-39f7-4faa-933a-3ff659c5b2a0)

![Screenshot 2024-03-24 at 8 04 53 PM](https://github.com/bcchang83/Naive-Bayes-Classifier-with-Breast-Cancer-Dataset/assets/54743478/18388f91-611e-4a86-b085-9a87cd6c3f72)

Both are comparable in the first experiment.

The second experiment, I tried to use linear regression to get missing data, and also compared the result of features selection with the result full features.

![Screenshot 2024-03-24 at 8 06 56 PM](https://github.com/bcchang83/Naive-Bayes-Classifier-with-Breast-Cancer-Dataset/assets/54743478/34a71cb1-05c4-4924-8ae7-830a57ebb549)

![Screenshot 2024-03-24 at 8 07 08 PM](https://github.com/bcchang83/Naive-Bayes-Classifier-with-Breast-Cancer-Dataset/assets/54743478/721a2b51-9f03-4b6a-8ce6-c5b4657218ee)

The features selection case is not improved, but the full features case is improved from 95.8% to 97%.
