from flask import Flask,render_template,flash, redirect,url_for,session,logging,request, make_response
from flask_sqlalchemy import SQLAlchemy

import sys
import csv
import collections
import time
import pandas as pd

app = Flask(__name__)

app.config.from_object('config')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)


class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    iduser = db.Column(db.Integer)
    username = db.Column(db.String(80))
    password = db.Column(db.String(80))

@app.route('/')
@app.route('/index/')
@app.route('/home/')
def index():
    userLogged = request.cookies.get('username')
    return render_template('index.html', userLogged = userLogged)

@app.route('/_rate')
def rate():
    id = request.args.get("id")
    rating = request.args.get("rating")
    ratingValues = [-1, -0.5, 0, 0.5, 1]
 
    idUser = float(request.cookies.get('idUser'))
    df = pd.read_csv('data/web_input.csv')
    
    for index, row in df.iterrows():
        if row["USER_ID"] == idUser:
            break
    jokeToUpdate = "Joke_" + id
    rating = int(rating) - 1

    df.at[index, jokeToUpdate] = ratingValues[rating]
    df.at[index, 'NEED_TO_CHANGE'] = 1

    df.to_csv("data/web_input.csv", index=False)

    return "success"

@app.route('/random/')
def random():
    try:
        voted = checkUserVoted()
        jokesList = collections.OrderedDict()
        if voted == 1:
            time.sleep(1)
            res = extractResults()
            for i in range (0, len(res)):
                fileToLoad = str(int(res[i]) + 1)
                source = 'jokes/init' + fileToLoad + '.html'     
                document = open(source,'r')
                content = document.read()
                content = content.replace('\n', '<br>')
                jokesList[res[i]] = content          
            userLogged = request.cookies.get('username')
            return render_template('random.html', jokesList = list(jokesList.items())[:10], userLogged = userLogged)
        else:
            for i in range (1,100):
                source = 'jokes/init' + str(i) + '.html'
                document = open(source,'r')
                content = document.read()
                content = content.replace('\n', '<br>')
                jokesList[i] = content
            userLogged = request.cookies.get('username')
            return render_template('random.html', jokesList = list(jokesList.items())[:10], userLogged = userLogged)
    except Exception, e:
        return str(e)    

@app.route('/login/',methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]

        login = user.query.filter_by(username=uname, password=passw).first()
        if login is not None:
            resp = make_response(render_template('index.html', userLogged = uname))
            resp.set_cookie('username', uname)
            return resp    
    userLogged = request.cookies.get('username')
    return render_template('login.html', userLogged = userLogged)

@app.route('/logout/',methods=["GET", "POST"])
def logout():
    resp = make_response(render_template('login.html'))
    resp.set_cookie('username', '')
    return resp   

@app.route('/register/',methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        passw = request.form['passw']
        exists = db.session.query(user.username).filter_by(username=uname).scalar() is not None
        if exists:
            return render_template('register.html', alreadyExists = True)
        sys.stdout.flush()
        newUserID = insertUserDefault()
        register = user(username = uname, password = passw, iduser = newUserID)
        db.session.add(register)
        db.session.commit()
        resp = make_response(redirect('/login'))
        resp.set_cookie('idUser', str(newUserID))
        return resp
    return render_template('register.html')

def insertUserDefault():
    with open('data/web_input.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        idList = list()
        next(reader)
        for i in reader:
            idList.append(int(i[0]))
        idNewUser = (max(idList) + 1)
        prepareNewRow = list()
        prepareNewRow.append(idNewUser)
        prepareNewRow.extend(['0', '50.0'])
        for nbJokes in range (0, 100):
            prepareNewRow.append('99.0')
    with open('data/web_input.csv','a') as fd:
        writer = csv.writer(fd)
        writer.writerow(prepareNewRow)
    return idNewUser

def checkUserVoted():
    idUser = request.cookies.get('idUser')
    df = pd.read_csv('data/web_input.csv')
    
    for index, row in df.iterrows():
        if row["USER_ID"] == str(idUser):
            break
    return df.at[index, 'NEED_TO_CHANGE']

def extractResults():
    idUser = request.cookies.get('idUser')
    
    results = list()
    with open('results/web.csv','r') as document:
        line = document.readline()
        while line:
            splited = line.split(',')
            if float(splited[0]) == float(idUser):
                break
            line = document.readline()
    for i in range (1, 11):
        joke = splited[i]
        results.append(joke[5:])

    return results

if __name__ == "__main__":
    db.create_all()
    app.run()
