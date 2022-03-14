## Dialect Detection 🗣️
Many countries speak Arabic; however, each country has its own dialect, the aim of this project is to
build a model that predicts the dialect given the text.


## Requirements 🐍
- [x] tensorflow==2.8.0
- [x] keras==2.8.0
- [x] fastapi==0.75.0
- [x] scikit-learn==1.0.2





## Dataset 📁

[![Downloads](https://img.shields.io/badge/Download-Data-blue)](https://drive.google.com/file/d/1uOFxMUprRFy-ruxCy_Hgnj9tdHoJkCc1/view?usp=sharing)

A dataset which has 2 columns, id and dialect. Target label column is the “dialect”, which has 18 classes. The “id” column will be used to retrieve the text, to do that, you need to call this API by a
POST request. The request body must be a JSON as a list of strings, and the size of the list must NOT
exceed 1000. The API will return a dictionary where the keys are the ids, and the values are the text, here
is a request and response sample.

Request body sample:
[ "
1055620304465215616", "1057418989293485952"
]

Response sample:

{

," في طريق مطروح مركز بهيج والمركز الي الي جمبه اسمه ايه "1055620304465215616": "@MahmoudWaked7 @maganenoo
"1057418989293485952": "@mycousinvinnyys @hanyamikhail1 "متهيالي دي شكولاته الهالوين فين المحل ده

}




## Model used 🤖

### Machine Learning Model
I used TfidfVectorizer and LinearSVC to train a classifier model to classify the Dialect.
### Why LinearSVC?
Fit better and faster than other models.

### Deep Learning Model
I used LSTM to train a classifier model to classify the Dialect.
### Why LSTM?
Fit better and faster than other models.


## Api! 🌍
To run the api simply clone the repository and download the weights files and then run the ```run.sh``` file. 



## Contact Me! 📢
If you have any queries or concerns, please feel free to contact me.
[LinkedIn](https://www.linkedin.com/in/ahmed-ayman-fawzy/)
