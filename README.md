# NaturalistAI: Mushroom Identification and Classification

![Mushrooms](https://github.com/jacobserfaty/Mushroom-Identification-Project/blob/main/Images/Mushrooms.png)

## Company Goals
- **Create a Mushroom Database**: Our goal is to develop a comprehensive database that catalogs mushrooms based on their morphology, habitat, season, and location.

- **Mushroom Classifier**: We aim to pair the database with a powerful classifier that can accurately identify edible mushrooms from poisonous ones, ensuring user safety.

- **User-Friendly Application**: We're working on creating an intuitive application that allows users to identify mushrooms in the wild by leveraging the classifier and the extensive mushroom database.

![Mushrooms2](https://github.com/jacobserfaty/Mushroom-Identification-Project/blob/main/Images/Mushrooms2.png)


## Mushroom Data Review

- **Mushroom Characteristics**: Mushrooms are typically brown, found in wooded areas, and thrive during the Summer and Autumn seasons.

- **Key Features**: Distinguishing between poisonous and edible mushrooms primarily relies on the morphology of the mushroom and, more crucially, the combination of morphological features.


## Classification Models: Assessment

- **Precision Metric**: To evaluate our models, we use precision as the primary metric. Precision gives more weight to false negatives, a critical factor when dealing with identifying poisonous mushrooms as edible.


## Classification Models: Evolution

### Initial Model

- The initial logistic regression model, without tuning, achieved a precision of 0.84.

![RegConMat](https://github.com/jacobserfaty/Mushroom-Identification-Project/blob/main/Images/RegConMat.png)

### Second Model: Feature Selection

- The second model, similar to the first, involved feature selection based on high multicollinearity.
- Some removed features included ring type (none), stem color (brown), stem color (white), and cap diameter.
- Surprisingly, this model performed worse than the original.

### Final Model: Decision Tree

- The final model resulted from extensive hyperparameter tuning, leading to a decision tree model.
- Key features in this model were cap shape, cap surface, cap color, and gill attachment.
- The final model achieved an impressive precision of 0.99 for both training and test data.

![DTConMat](https://github.com/jacobserfaty/Mushroom-Identification-Project/blob/main/Images/DTConMat.png)

- The results presented on the ROC curve graph show that the final model has achieved almost perfect accuracy.

![ROCCurve](https://github.com/jacobserfaty/Mushroom-Identification-Project/blob/main/Images/ROCCurve.png)


## Business Recommendations

- **Mushroom Identification App**: Develop a user-friendly application that empowers users to identify mushrooms based on their unique features.

- **Mushroom Catalog**: Create a comprehensive catalog of mushrooms, including detailed feature information, allowing users to search and learn about different mushrooms.

- **User Contributions**: Enable users to contribute images and features of mushrooms they encounter in the wild, continually improving the database and the model's accuracy.


## Future Improvements

- **Enhanced Classification**: Combine the feature classification model with an image classification model, such as a convolutional neural network (CNN), to improve accuracy.

- **Detailed Features**: Expand the database with specific features for each mushroom, including habitat, tree type they grow under, preference for dead or living trees, ground vs. tree growth, clustering behavior, and more.
