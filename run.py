#! /usr/bin/env python

import biapp
from biapp import app

if __name__ == "__main__":
    app.secret_key = "ABCDEFF"
    app.run(debug=True)
