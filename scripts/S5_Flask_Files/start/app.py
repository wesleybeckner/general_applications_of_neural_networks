from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)

@app.route('/form/')
def index():
   return render_template("index.html")

@app.route('/static/')
def index2():
   return render_template("static.html")

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login', methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))

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

@app.route('/<somevariable>/<anotherone>')
def hello_world(somevariable, anotherone):
    return "first word: {}\nsecond word: {}!".format(somevariable, anotherone)

@app.route('/type/<int:var>')
def check_type(var):
    print(type(var))
    return "the type is: {}".format(type(var).__name__)

@app.route('/type/<string:var>')
def check_type2(var):
    print(type(var))
    return "the type is: {}".format(type(var).__name__)

@app.route('/type/<float:var>')
def check_type3(var):
    print(type(var))
    return "the type is: {}".format(type(var).__name__)

# def hello_world():
#    return "hello world"
# app.add_url_rule("/", view_func=hello_world)

# def hello_world(msg):
#    return "{}!".format(msg)
# app.add_url_rule("/<msg>", view_func=hello_world)

if __name__ == '__main__':
    app.run(debug=False)