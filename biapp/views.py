from flask import Flask,render_template,flash, redirect,url_for,session,logging,request
from flask_sqlalchemy import SQLAlchemy

import sys

app = Flask(__name__)

app.config.from_object('config')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    password = db.Column(db.String(80))

@app.route('/')
@app.route('/index/')
@app.route('/home/')
def index():
    return render_template('index.html')

@app.route('/random/')
def random():
    return render_template('random.html')

@app.route('/login/',methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]

        login = user.query.filter_by(username=uname, password=passw).first()
        if login is not None:
            return redirect(url_for("index"))    
    return render_template('login.html')

@app.route('/register/',methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        passw = request.form['passw']

        print(uname, file=sys.stdout)
        print(passw, file=sys.stderr)
        sys.stdout.flush()
        register = user(username = uname, password = passw)
        db.session.add(register)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template('register.html')

if __name__ == "__main__":
    db.create_all()
    app.run()
