<a href="https://colab.research.google.com/github/wesleybeckner/technology_fundamentals/blob/main/C4%20Machine%20Learning%20II/LABS_PROJECT/Tech_Fun_C4_L2_Testing_and_Serving_Code.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Technology Fundamentals Course 4, Lab 2: DevOps: Testing and Serving Code.ipynb

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

**Teaching Assitants**: Varsha Bang, Harsha Vardhan

**Contact**: vbang@uw.edu, harshav@uw.edu
<br>

---

<br>

In this lab we will practice writing unit tests (part 1) as well as serving our python code in a web framework (part 2). There is an optional part 3 where we move our unit tests into a local directory and run them with `pytest`.

<br>

---

<img src="https://www.pentalog.com/wp-content/uploads/2020/03/DevOps-engineer-job-roles-and-responsibilities.png"></img>




# Part 1: Writing Tests


```python
!pip install fastapi
```

    Collecting fastapi
    [?25l  Downloading https://files.pythonhosted.org/packages/4e/b9/a91a699f5c201413b3f61405dbccc29ebe5ad25945230e9cec98fdb2434c/fastapi-0.65.1-py3-none-any.whl (50kB)
    [K     |████████████████████████████████| 51kB 6.8MB/s 
    [?25hCollecting pydantic!=1.7,!=1.7.1,!=1.7.2,!=1.7.3,!=1.8,!=1.8.1,<2.0.0,>=1.6.2
    [?25l  Downloading https://files.pythonhosted.org/packages/9f/f2/2d5425efe57f6c4e06cbe5e587c1fd16929dcf0eb90bd4d3d1e1c97d1151/pydantic-1.8.2-cp37-cp37m-manylinux2014_x86_64.whl (10.1MB)
    [K     |████████████████████████████████| 10.1MB 35.9MB/s 
    [?25hCollecting starlette==0.14.2
    [?25l  Downloading https://files.pythonhosted.org/packages/15/34/db1890f442a1cd3a2c761f4109a0eb4e63503218d70a8c8e97faa09a5500/starlette-0.14.2-py3-none-any.whl (60kB)
    [K     |████████████████████████████████| 61kB 9.2MB/s 
    [?25hRequirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from pydantic!=1.7,!=1.7.1,!=1.7.2,!=1.7.3,!=1.8,!=1.8.1,<2.0.0,>=1.6.2->fastapi) (3.7.4.3)
    Installing collected packages: pydantic, starlette, fastapi
    Successfully installed fastapi-0.65.1 pydantic-1.8.2 starlette-0.14.2



```python
import random
import numpy as np
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
```

## Types of Tests

There are two main types of tests we want to distinguish:
* **_Unit test_**: an automatic test to test the internal workings of a class or function. It should be a stand-alone test which is not related to other resources.
* **_Integration test_**: an automatic test that is done on an environment, it tests the coordination of different classes and functions as well as with the running environment. This usually precedes sending code to a QA team.

To this I will add:

* **_Acid test_**: extremely rigorous tests that push beyond the intended use cases for your classes/functions. Written when you, like me, cannot afford QA employees to actually test your code. (word origin: [gold acid tests in the 1850s](https://en.wikipedia.org/wiki/Acid_test_(gold)), [acid tests in the 70's](https://en.wikipedia.org/wiki/Acid_Tests))

In this lab we will focus on _unit tests_.

## Unit Tests

Each unit test should test the smallest portion of your code possible, i.e. a single method or function. Any random number generators should be seeded so that they run the exact same way every time. Unit tests should not rely on any local files or the local environment. 

Why bother with Unit Tests when we have Integration tests?

A major challenge with integration testing is when an integration test fails. It’s very hard to diagnose a system issue without being able to isolate which part of the system is failing. Here comes the unit test to the rescue. 

Let's take a simple example. If I wanted to test that the sume of two numbers is correct


```python
assert sum([2, 5]) == 7, "should be 7"
```

Nothing is sent to the print out because the condition is satisfied. If we run, however:

```
assert sum([2, 4]) == 7, "should be 7"
```

we get an error message:

```
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
<ipython-input-3-d5724b127818> in <module>()
----> 1 assert sum([2, 4]) == 7, "should be 7"

AssertionError: should be 7
```


To make this a Unit Test, you will want to wrap it in a function


```python
def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"

test_sum()
print("Everything passed")
```

    Everything passed


And if we include a test that does not pass:

```
def test_sum():
  assert sum([3, 3]) == 6, "Should be 6"

def test_my_broken_func():
  assert sum([1, 2]) == 5, "Should be 5"

test_sum()
test_my_broken_func()
print("Everything passed")
```



Here our test fails, because the sum of 1 and 2 is 3 and not 5. We get a traceback that tells us the source of the error:

```
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
<ipython-input-13-8a552fbf52bd> in <module>()
      6 
      7 test_sum()
----> 8 test_my_broken_func()
      9 print("Everything passed")

<ipython-input-13-8a552fbf52bd> in test_my_broken_func()
      3 
      4 def test_my_broken_func():
----> 5   assert sum([1, 2]) == 5, "Should be 5"
      6 
      7 test_sum()

AssertionError: Should be 5
```



Before sending us on our merry way to practice writing unit tests, we will want to ask, what do I want to write a test about? Here, we've been testing sum(). There are many behaviors in sum() we could check, such as:

* Does it sum a list of whole numbers (integers)?
* Can it sum a tuple or set?
* Can it sum a list of floats?
* What happens if one of the numbers is negative? etc..

In the end, what you test is up to you, and depends on your intended use cases. As a general rule of thumb, your unit test should test what is relevant.

The only caveat to that, is that many continuous integration services (like [TravisCI](https://travis-ci.com/)) will benchmark you based on the percentage of lines of code you have that are covered by your unit tests (ex: [85% coverage](https://github.com/wesleybeckner/gains)).

## L2 Q1 Write a Unit Test

Remember our Pokeball discussion in C2? We'll return to that here. This time writing unit tests for our classes.

Sometimes when writing unit tests, it can be more complicated than checking the return value of a function. Think back on our pokemon example:

<br>

<p align=center>
<img src="https://cdn2.bulbagarden.net/upload/thumb/2/23/Pok%C3%A9_Balls_GL.png/250px-Pok%C3%A9_Balls_GL.png"></img>

```
class Pokeball:
  def __init__(self, contains=None, type_name="poke ball"):
    self.contains = contains
    self.type_name = type_name
    self.catch_rate = 0.50 # note this attribute is not accessible upon init

  # the method catch, will update self.contains, if a catch is successful
  # it will also use self.catch_rate to set the performance of the catch
  def catch(self, pokemon):
    if self.contains == None:
      if random.random() < self.catch_rate:
        self.contains = pokemon
        print(f"{pokemon} captured!")
      else:
        print(f"{pokemon} escaped!")
        pass
    else:
      print("pokeball is not empty!")
      
  def release(self):
    if self.contains == None:
      print("Pokeball is already empty")
    else:
      print(self.contains, "has been released")
      self.contains = None
```

If I wanted to write a unit test for the release method, I couldn't directly check for the output of a function. I'll have to check for a **_side effect_**, in this case, the change of an attribute belonging to a pokeball object; that is the change to the attribute _contains_.




```python
class Pokeball:
  def __init__(self, contains=None, type_name="poke ball"):
    self.contains = contains
    self.type_name = type_name
    self.catch_rate = 0.50 # note this attribute is not accessible upon init

  # the method catch, will update self.contains, if a catch is successful
  # it will also use self.catch_rate to set the performance of the catch
  def catch(self, pokemon):
    if self.contains == None:
      if random.random() < self.catch_rate:
        self.contains = pokemon
        print(f"{pokemon} captured!")
      else:
        print(f"{pokemon} escaped!")
        pass
    else:
      print("pokeball is not empty!")
      
  def release(self):
    if self.contains == None:
      print("Pokeball is already empty")
    else:
      print(self.contains, "has been released")
      self.contains = None
```

In the following cell, finish the code to test the functionality of the _release_ method:


```python
def test_release():
  ball = Pokeball()
  ball.contains = 'Pikachu'
  ball.release()
  # turn the pseudo code below into an assert statement
  
  ### YOUR CODE HERE ###
  # assert <object.attribute> == <something>
```


```python
test_release()
```

    Pikachu has been released


## L2 Q2 Write a Unit Test for the Catch Rate

First, we will check that the succcessful catch is operating correctly. Remember that we depend on `random.random` and condition our success on whether that random value is less than the `catch_rate` of the pokeball:

```
if self.contains == None:
      if random.random() < self.catch_rate:
        self.contains = pokemon
```

so to test whether the successful catch is working we will seed our random number generator with a value that returns less than the `catch_rate` of the pokeball and then write our assert statement:



```python
def test_successful_catch():
  # choose a random seed such that
  # we know the catch call should succeed
  
  ### YOUR CODE BELOW ###
  # random.seed(<your number here>)
  ball = Pokeball()
  ball.catch('Psyduck') # Sabrina's fave pokemon

  ### YOUR CODE BELOW ###
  # <object.attribute> == <something>, "ball did not catch as expected"
```

NICE. Now we will do the same thing again, this time testing for an unsuccessful catch. SO in order to do this, we need to choose a random seed that will cause our catch to fail:


```python
def test_unsuccessful_catch():
  # choose a random seed such that
  # we know the catch call should FAIL
  
  ### YOUR CODE BELOW ###
  # random.seed(<your number here>)
  ball = Pokeball()
  ball.catch('Psyduck') # Sabrina's fave pokemon

  ### YOUR CODE BELOW ###
  # <object.attribute> == <something>, "ball did not fail as expected"
```

When you are finished test your functions below


```python
test_unsuccessful_catch()
```

    Psyduck captured!



```python
test_successful_catch()
```

    Psyduck captured!


## L2 Q3 Write a Unit Test that Checks Whether the Overall Catch Rate is 50/50

For this one, we're going to take those same ideas around seeding the random number generator. However, here we'd like to run the catch function multiple times to check whether it is truly creating a 50/50 catch rate situation.

Here's a pseudo code outline:

1. seed the random number generator
2. for 100 iterations: 
  * create a pokeball
  * try to catch something
  * log whether it was successful
3. check that for the 100 attempts the success was approximately 50/50

_note:_ you can use my `suppress stdout()` function to suppress the print statements from `ball.catch`

ex:

```
with suppress_stdout():
  print("HELLO OUT THERE!")
```

> quick segway: what is the actual behavior of `random.seed()`? Does it produce the same number every time we call `random.random()` now? Check for yourself:


```python
random.seed(42)
[random.random() for i in range(5)]
```




    [0.6394267984578837,
     0.025010755222666936,
     0.27502931836911926,
     0.22321073814882275,
     0.7364712141640124]



We see that it still produces random numbers with each call to `random.random`. However, those numbers are the same with every execution of the cell. What happens when we do this:


```python
[random.random() for i in range(5)]
```




    [0.6766994874229113,
     0.8921795677048454,
     0.08693883262941615,
     0.4219218196852704,
     0.029797219438070344]



The numbers are different. BUT:


```python
random.seed(42)
[random.random() for i in range(10)]
```




    [0.6394267984578837,
     0.025010755222666936,
     0.27502931836911926,
     0.22321073814882275,
     0.7364712141640124,
     0.6766994874229113,
     0.8921795677048454,
     0.08693883262941615,
     0.4219218196852704,
     0.029797219438070344]



We see them here in the bottom half of the list again. So, random.seed() is _seeding_ the random number generator such that it will produce the same sequence of random numbers every time, from the given seed. This will reset whenever random.seed() is set again. This behavior is useful because it allows us to continue using random number generation in our code, (for testing, creating examples and demos, etc.) but it will be reproducable each time.

_End Segway_


```python
def test_catch_rate():
  ### YOUR CODE HERE ###
  
  ### END YOUR CODE ###
  assert np.abs(np.mean(results) - 0.5) < 0.1, "catch rate not 50/50"
test_catch_rate()
```

## Test Runners

When we start to create many tests like this, it can be cumbersome to run them all at once and log which ones fail. To handle our unit tests we use what are called **_test runners_**. We won't dedicate time to any single one here but the three most common are:

* unittest
* nose2
* pytest

unittest is built into python. I don't like it because you have to follow a strict class/method structure when writing the tests. nose2 is popular with many useful features and is generally good for high volumes of tests. My favorite is pytest, it's flexible and has an ecosystem of plugins for extensibility. 


```python
# maybe have a demo of writing a file from jupyterlab cell
# and then running that test file with pytest

# conversely, could go to the actual command line since it looks like
# everyone has a local environment, have them clone a few files from truffletopia
# and demo pytest that way.

!pytest test_release
```

# Part 2: Serving Python

Our next objective is to serve our code to the wide, wide world (ahem, the world, wide, web) in as simple a manner as possible. As _s i m p l e_ as possible. 


```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Copy the above code into a local file `main.py` 

install fastapi and uvicorn with:

```
pip install fastapi[all]
```

then from the terminal run:

```
uvicorn main:app --reload
```

`uvicorn` is the server we will use to run our fastapi application. `main` refers to the name of the file to run and `app` the object within it. `--reload` will cause the server to reboot the app anytime changes are made to the file `main.py`

You should see on the command line now something like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

This is telling us where our python app is running 

## Interactive API docs

Now go to http://127.0.0.1:8000/docs



## Recap, step by step

1. we imported fastapi

`from fastapi import FastAPI`

2. created a `FastAPI` instance

`app = FastAPI()` 

3. created a _path_ _operation_

**_path_** here refers to the last part of the URL starting from the first `/`. So in a URL like:

`truffletopia.io/basecake/chiffon`

...the path would be:

`/basecake/chiffon`

> a path is commonly referred to as an "endpoint" as in "API endpoint" or a "route"

**_operation_** refers to one of the HTTP "methods"

One of:

* `POST`: create data
* `GET`: read data
* `PUT`: update data
* `DELETE`: delete data

...and more exotic ones

We can think of these HTTP methods as synonymous with _operation_. Taking it together:

`@app.get("/")`

tells FastAPI that the function right below is in charge of handling requests that go to:

* the path `/`
* using a `get` operation

> the `@` in python is called a decorator and lets the python executor know it is going to be modifying a function in some way, in this case FastAPI's handling of the `get` requests to `/`

4. define the **_path operation function_**

* path is `/`
* operation is `GET`
* function is the funtion below the decorator

If you're curious about the `async` infront of our path operation function you can read about it [here](https://fastapi.tiangolo.com/async/#in-a-hurry).
