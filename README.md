# Predicting Cy Young Award Votes Based on MLB Pitching Statistics

### Aidan Murphy, Luke VanHouten
### CSE 163

## Context and Setup
Our project is to predict the amount of Cy Young Award points that a top-tier MLB pitcher receives based on their pitching statistics. The primary place that the data that we will be using for our analysis can be found is from the pybaseball module for Python. This can be installed by running the following command in the command prompt:

```bash
pip install pybaseball
```

Next, the other datasource that will need to be downloaded is a csv file of MLB player ID's called `players.csv`. There is a link to this in the report deliverable for this project, but [here is another link to it just in case](https://mega.nz/file/qQs21QjL#QuogPGa7zuexuc37QeeyBqDfTNNsNQGOnEJkK0zyXCI). This file will also be submitted alongside this README as well as the rest of the files from the project. You will need to know what the directory of this file is on your computer when you run the program.

## Running the Model
Nothing in the data file `baseball_data.py` needs to be ran, as this is all accessed by the model file. Its class name from this file is `PitchingData`. In order to run the model file, you will need to to change the values of the variables of `start_date`, `end_date`, and `id_filename` in the main method in the `baseball_model.py`. The date variables correspond to the start and end date of the analysis that is to be performed. **Ensure that the value of `start_date` is less than the value of `end_date`.** The acceptible date range for this is between 1956 and 2016, which what the results of the models are taken from. Do note that this may take a very long time to run. The variable for `id_filename` corresponds to the directory filename of the `players.csv` file downloaded from before. The class name for this file is called `PitchingModel`. 

**When running the `compare_eras()` method, make sure that the year 1990 is included in your date range!**

## Running the Testing File

Just like with the model file, the testing file `test_baseball.py` will need to be run in a similar way. As it calls both of the previous files, the same inputs will be needed to used for both calls. The inputs for the `PitchingData` class is exactly the same as those for the `PitchingModel` class. The testing files rely on assertion errors. In order to run this file, you will need to set the value of `id_filename` to whatever your file path is for the `players.csv` file and then run the script. We have provided a file `cse163_utils.py` that includes a helper function to check assertion equal statements. All of the Python scripts discussed in this README will need to be loacted in the same directory.