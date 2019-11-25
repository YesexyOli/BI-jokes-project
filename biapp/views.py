from flask import Flask,render_template,flash, redirect,url_for,session,logging,request, make_response
from flask_sqlalchemy import SQLAlchemy

import sys
import csv

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

@app.route('/random/')
def random():
    try:
        jokesList = list()
        for i in range (1,100):
            source = 'jokes/init' + str(i) + '.html'
            document = open(source,'r')
            content = document.read()
            content = content.replace('\n', '<br>')
            jokesList.append(content)
        userLogged = request.cookies.get('username')
        return render_template('random.html', jokesList = jokesList[:10], userLogged = userLogged)
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
        # print(uname, sys.stdout)
        # print(passw, sys.stderr)
        sys.stdout.flush()
        newUserID = insertUserDefault()
        register = user(username = uname, password = passw, iduser = newUserID)
        db.session.add(register)
        db.session.commit()
        return redirect(url_for("login"))
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
        prepareNewRow.extend(['1', '50.0'])
        for nbJokes in range (0, 100):
            prepareNewRow.append('99.0')
    with open('data/web_input.csv','a') as fd:
        writer = csv.writer(fd)
        writer.writerow(prepareNewRow)
    return idNewUser

if __name__ == "__main__":
    db.create_all()
    app.run()
