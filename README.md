# BI-jokes-project

Find the best jokes for you thanks to a recommendation system algorithm.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install flask.
(You may need to use sudo)
The Server script works with python 2 
The recommander script works with python 3
```
pip2 install flask
pip2 install Flask-SQLAlchemy
pip2 install pandas 
pip install pandas
pip install --upgrade tqdm
pip install numpy
```

## Usage

```
python2 run.py
python prediction_web.py data/web_input.csv data/centers_84.csv results/web.csv 3 1 True
```

## find your IP_Adress (Linux )

```
 hostname -I
```
## access website 
```
{your_IP_Adress}:5000/ 
192.168.0.8:5000 as an example
```
If you still have Problems also change configuration as described in  http://dixu.me/2015/10/26/How_to_Allow_Remote_Connections_to_Flask_Web_Service/
