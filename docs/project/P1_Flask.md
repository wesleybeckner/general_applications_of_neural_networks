<a href="https://colab.research.google.com/github/wesleybeckner/general_applications_of_neural_networks/blob/main/notebooks/project/P1_Flask.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# General Applications of Neural Networks <br> P1: The Flask Application Factory

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

<br>

---

<br>


In this session we'll borrow heavily from [hackers and slackers](https://hackersandslackers.com/flask-application-factory/)!

<p align="center">
<img src="https://raw.githubusercontent.com/wesleybeckner/oonii/main/assets/flask_factory.jpg" width=700px></img>
</p>

<br>

---

<br>


## Setup

Before we start...

1. Download and install Anaconda
2. Download and install VS Code
3. Clone this repository

## Core Technologies
[original post](https://www.tutorialspoint.com/flask/flask_quick_guide.htm)

### What is Web Framework?
Web Application Framework or simply Web Framework represents a collection of libraries and modules that enables a web application developer to write applications without having to bother about low-level details such as protocols, thread management etc.

### What is Flask?
Flask is a web application framework written in Python. It is developed by Armin Ronacher, who leads an international group of Python enthusiasts named Pocco. Flask is based on the Werkzeug WSGI toolkit and Jinja2 template engine. Both are Pocco projects.

### WSGI
Web Server Gateway Interface (WSGI) has been adopted as a standard for Python web application development. WSGI is a specification for a universal interface between the web server and the web applications.

### Werkzeug
It is a WSGI toolkit, which implements requests, response objects, and other utility functions. This enables building a web framework on top of it. The Flask framework uses Werkzeug as one of its bases.

### Jinja2
Jinja2 is a popular templating engine for Python. A web templating system combines a template with a certain data source to render dynamic web pages.

Flask is often referred to as a micro framework. It aims to keep the core of an application simple yet extensible. Flask does not have built-in abstraction layer for database handling, nor does it have form a validation support. Instead, Flask supports the extensions to add such functionality to the application. Some of the popular Flask extensions are discussed later in the tutorial.

## The Application Factory

Eventually we will want to structure our application as follows:

```
/app
├── /application
│   ├── __init__.py
│   ├── auth.py
│   ├── forms.py
│   ├── models.py
│   ├── routes.py
│   ├── /static
│   └── /templates
├── config.py
└── wsgi.py
```

This is our directory structure for creating an [application factory](https://flask.palletsprojects.com/en/1.0.x/patterns/appfactories/) in Flask. 
But for now, we are going to take a much simpler approach at the expense of creating something well-structured.

> Notice how there isn't even a file called `app.py`!


## Create app.py



```python
## btw, we can run flask right in our jupyter notebook:

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello World!"

if __name__ == '__main__':
    app.run()
```

     * Serving Flask app '__main__' (lazy loading)
     * Environment: production
    [31m   WARNING: This is a development server. Do not use it in a production deployment.[0m
    [2m   Use a production WSGI server instead.[0m
     * Debug mode: off


     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)


### 💪 Exercise 1

Create a file called `app.py` and paste in the following:

```
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
   return "Hello World!"

if __name__ == '__main__':
   app.run()
```

What are we doing here? Let's break it down piecewise. 

* `from flask import Flask`

The Flask blueprint is going to contain the bones of our WSGI application. Let's see how we envoke the object:

* `app = Flask(__name__)`

Prototypical flask styling is to use `app` as the namespace for the Flask object. The class blueprint take the current module name, `__name__` as argument. 

* `@app.route('/')`

The `@` operates as a decorator, meaning we are going to be altering something predefined in the Flask class. In this case, we are telling our application what to execute when the server is accessed at a specific url. In this case, the base,`/`, url

* `def hello_world():`
* `  return "Hello World!"`

Our function for handling the route appears directly below `app.route('/')`, this will be the case for any route we wish to define (`@` decorator followed by function definition). In our simple, starting app, we pass the obligatory phrase, "Hello World!"

* `if __name__ == '__main__':`

Only run our application if it is not imported into the python interpreter by a previously running script

* `app.run()`

Finally, we run our application. A full description of the parameters we can pass to `app.run()`, all are optional:

| Location | Parameter | Description                                                                                                    |
|----------|-----------|----------------------------------------------------------------------------------------------------------------|
| 1        | host      | Hostname to listen on. Defaults to 127.0.0.1 (localhost). Set to '0.0.0.0' to have server available externally |
| 2        | port      | Defaults to 5000                                                                                               |
| 3        | debug     | Defaults to False. If set to True, provides debug info and causes application to restart with code changes                                                         |
| 4        | options   | To be forwarded to WSGI server                                                                        |

## Routing

```
@app.route('/')
def hello_world():
   return "Hello World!"
```

Routing handles what is generated to the web browser (or returned in an API request) when the user accesses a specific url. The `@` decorator is what is typically envoked but we can also use the function `add_url_rule`. 




```python
app.add_url_rule
```




    <bound method Flask.add_url_rule of <Flask '__main__'>>



### 💪 Exercise 2:

Let's try changing the `app.py` file to this instead:

```
def hello_world():
   return "hello world"
app.add_url_rule("/", view_func=hello_world)
```

## Variable Rules

Frequently, especially when we get into build flask-based APIs, we will want to include variable parameters in the url path. we can do this by designating `<variable-name>` in the url.

### 💪 Exercise 3:

extend your url path to include a new variable, i.e. `/<msg>` and return the message from the path to the screen. Ex:

```
def hello_world(msg):
   return "{}!".format(msg)
app.add_url_rule("/<msg>", view_func=hello_world)
```

Note that when you type the url in the browser the carrot brackets should be omitted. 

We can also specify the datatype passed for the variable

* integer: `<int:var-name>`
* float: `<float:var-name>`


```python
def hello_world(msg):
    return "{}!".format(msg)
app.add_url_rule("/<msg>", view_func=hello_world)
```

### 💪 Exercise 4:

experiment with datatype enforcement by making a new url path called `/type/<string:var>` and passing to it in 3 different cases: int, string, and float, and return the type to the browser, i.e:

`return "the type is: {}".format(type(var).__name__)`

what happens when you try to pass an _int_ to _float_ or a _float_ to _int_?

Ex:

```
@app.route('/type/<int:var>')
def check_type(var):
   print(type(var))
   return "the type is: {}".format(type(var).__name__)
```


```python
@app.route('/type/<int:var>')
def check_type(var):
    print(type(var))
    return "the type is: {}".format(type(var).__name__)
```




    "<class 'float'>"



## Dynamic URL Building

url redirects can be initiated depending on input variables. For example, imagine you have a `/user/<name>` url that redirects to `/guest/<name>` or `/admin/` depending on if the `<name>` variable is set to `admin` or not. You would invoke this like so:

```
@app.route('/user/<name>')
def hello_user(name):
   if name == 'admin':
      return redirect(url_for('hello_admin'))
   else:
      return redirect(url_for('hello_guest', name=name))

@app.route('/admin/')
def hello_admin():
   return "Hello Admin!"

@app.route('/guest/<name>')
def hello_guest(name):
   return "Hello {}!".format(name)
```

### 💪 Exercise 5:

paste the above code into `app.py` and test out the functionality by visiting `/user/<yourname>` and `user/admin`. Notice how the url is redirected in the web address bar!

## HTTP Methods

So far, we have been defaulting to the http method, GET. But there are other http methods we can envoke:

| Method | Description                                                                                 |
|--------|---------------------------------------------------------------------------------------------|
| GET    | Sends data in unencrypted form to the server                                                |
| HEAD   | Same as GET but without response body                                                       |
| POST   | Used to send HTML form data to server. Data received by POST method is not cached by server |
| PUT    | Replaces all current representations of the target resource with the uploaded content       |
| DELETE | Removes all current representations of the target resource given by a URL                   |

### 📝 💪 Exercise 6:

Under `/templates/` create a new file called `/index.html`/ and paste in the following:

```
<html>
   <body>
      <form action = "http://localhost:5000/login" method = "post">
         <p>Enter Name:</p>
         <p><input type = "text" name = "nm" /></p>
         <p><input type = "submit" value = "submit" /></p>
      </form>
   </body>
</html>
```

Import `request` from `flask` at the top of `app.py`.

Now paste the following into your `app.py` file:

```
@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))
      
```

the two lines we'll inspect are `user = request.form['nm']` and `user = request.args.get('nm')`. In `index.html` the method is set to `post`. In this case, when we open `index.html` in the web browser and fill the input object who's name is `nm` and click submit, it sends a post request to localhost:5000/login. The variable `user` in the flask application is then set according to `request.form['nm']`. Go ahead and try this out yourself.

After verifying that this works, change method to `"get"` in `index.html`. Now the user variable will be set with `request.args.get('nm')` in the flask application.

## Templates 
### (jinja2 and where html lives)

We can insert html styling directly in our python code, but that gets clunky real fast. Instead, we typically leverage the [jinja templating engine](https://jinja.palletsprojects.com/en/3.0.x/) to render html files according to routes specified in the flask application. Let's take an example

### 💪 Exercise 7:

import `render_template` from `flask` at the top of `app.py`. Then add an url rule at `/form/` that returns `render_template("login.html")`

Ex:

```
@app.route('/form/')
def index():
   return render_template("login.html")
```

When you visit `localhost:5000/form/` you should see the page rendered by `index.html` that we opened previously! In this way, you can build a python website that still utilizes any html/css/js you would like in a traditional static build.

> Note: Flask will always try to find your html files called via render_template in your templates/ folder sitting in the same directory as your python script

## Static Files
### (where CSS and JS live)

For this next piece, we're going to include 3 new files in our application and show how traditional html/css/js websites can be rendered in the Flask framework.

### 📝 💪  Exercise 8:

Copy the following into the respect files and locations.

In `templates/static.html`:

```
<head>
    <link rel="stylesheet" href="../static/static.css">
</head>

<body>
    <div class="flex-container">
        <div id="thumbnails">
            <h1>💪</h1>
            <h1>🍫</h1>
            <h1>⛱️</h1>
        </div>
    </div>
    <script src="../static/static.js"></script>
</body>
```

In `static/static.css`:

```
.flex-container {
    text-align:center;
  }
```

In `static/static.js`:

```
var thumbnails = document.getElementById("thumbnails")
var emoji = thumbnails.innerHTML.split("\n")
console.log(emoji)

emoji.sort(() => Math.random() - 0.5);

var str = ""
for (let i=0; i<emoji.length;i++){
  str += emoji[i] + "\n"
}
console.log(str)
document.getElementById("thumbnails").innerHTML = str
```

In our JS file we are randomly swapping the innerHTML settings of the different emojis (effectively randomizing the order of the emojis on the webpage) open `static.html` in the web browser and inspect to see the effect our of JS code, you can also look at the console to view the output of the randomization process. 

Now, using the same methodology as we did in exercise 7, add the route to `app.py` so that users will see `static.html` when they visit the url extension `emoji`.


