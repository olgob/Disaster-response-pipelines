# Disaster Response Pipeline Project

### Summary of the project

In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The disaster data contain real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that one can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

Below are a few screenshots of the web app.

<img src="https://github.com/olgob/Disaster-response-pipelines/blob/master/Screenshots/disaster_response_pipeline_home_page.jpg" width="800">

<img src="https://github.com/olgob/Disaster-response-pipelines/blob/master/Screenshots/disaster_response_pipeline_research_page.jpg" width="800">

### How to run the Python scripts and web app

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Find the workspace environmental variables with `env | grep WORK`, and you can open a new browser window and go to the address:
`http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` replacing WORKSPACEID and WORKSPACEDOMAIN with your values.

### Explanation of the files in the repository

<pre>
.
├── app
│   ├── run.py
|   ├── text_length_extractor.py
|   ├── wrangling.py
│   ├── templates
│       ├── go.html
│       ├── master.html
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   ├── process_data.py
├── screenshots
├── models
│   ├── text_length_extractor.py
│   ├── train_classifier.py

</pre>
