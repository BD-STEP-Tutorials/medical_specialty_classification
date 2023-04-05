# medical_specialty_classification tutorial project

## Motivation

- Adaptation of Kaggle competition: https://www.kaggle.com/competitions/medical-specialty-classification/
- These are publically available medical notes with corresponding specialties
- Machine learning goal is to predict the medical specialty from the content of the text of the note
- Tutorial goal is to introduce good practices in structuring, automating and documenting data science projects

## Codespace Setup

- Project is intended to be run from GitHub codespace
- Codespace should automatically install Python 3.10, Python package requirements, source code, scispacy nlp model and select VSCode Extensions when first built
- Full details of codespace build in .devcontainer/devcontainer.json file

## Raw Data

- Originally downloaded from https://www.kaggle.com/competitions/medical-specialty-classification/ 
- In the project, the raw data is stored at: data/raw/medical_text.csv

## Stages of Pipeline in src folder

1. Clean Raw Data (src/clean_data.py)
    - Removes medical_specialty categories with less than 50 records and general categories (e.g. "Progress Notes") 
    - Also makes a simplified (abbreviated) category name for each remaining category medical_specialty 
2. Text Cleaning/Preprocessing (src/process_text.py)
    - Uses scispacy model in nlp_models/en_core_sci_sm-0.5.1 to clean text and extract entities
    - Adds cleaned text cols 'text_clean' and 'text_clean_ents' to output csv
3. Split Data in Train and Test Data (src/split_data.py)
    - Uses train_test_split function from sklearn to split data 80/20 into training and test (holdout) data
4. Train and Tune Text Classifier model (src/train_classifier.py)
    - Transforms text to Word-Document matrix using TF-IDF
    - Trains and tunes hyperparameters of a RandomForestClassifier model using sklearn
    - Does 5-fold cross validation during tuning with RandomSearchCV and then refits model on all training data using parameters found to give best fit (using f1 score metric)
    - Stores best performing model in ml_models folder (ml_models/rfc_tfidf_tx_clean_ents.joblib)
5. Evaluate Performance of Model on Test Data (src/evaluate_classifier.py)
    - Evaluates model performance on test (holdout) data
    - Generates classification report on model performance that is stored in reports folder (reports/clf_report_rfc_tfidf_text_clean_ents.csv)
    - Generates confusion matrix plot that is stored in reports folder (reports/confusion_matrix_rfc_tfidf_text_clean_ents.png)

## makefile Details

- makefile (in project root folder) automates running of pipeline and project setup
- Specifically, when running pipeline, makefile invokes python commands from command line needed to build project targets (outputs)
- makefile is commented to give more details on each command
- makefile commands must be run from BASH terminal opened in project root folder

### Running Project

- Project can be run from end-to-end via makefile command run from project folder using:


        $ make all

- Individual steps of the pipeline can be run as indicated in the makefile. All project artifacts/outputs can be deleted using:


        $ make clean


- More detail on inputs/outputs and usage of stages can be obtained by running (from project root):

        $ python src/<name_of_.py_file> --help

    e.g.

        $ python src/clean_data.py --help

## EDA Folder

- Contains some EDA/development work done with iPython from VSCode used to explore data and develop some of the stages in the source code pipeline

## setup.py file

- Used to install project source code into Codespace Python environment

## LICENSE

- MIT License states how project software can be used (very permissive) 

## Author/Contact

- David Winski (David.Winski@va.gov)