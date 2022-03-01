from flask import Flask, render_template_string, request, make_response
from TicTacToe import *

TEXT = """
<!doctype html>
<html>
  <head><title>Tic Tac Toe</title></head>
  <body>
    <h1>Tic Tac Toe</h1>
    <h2>{{msg}}</h2>
    <form action="" method="POST">
      <table>
        {% for j in range(0, 3) %}
        <tr>
          {% for i in range(0, 3) %}
          <td>
            <button type="submit" name="choice" value="{{j*3+i+1}}"
              {{"enabled"}}>
              {{"X"}}
            </button>
          </td>
          {% endfor %}
        </tr>
        {% endfor %}
      </table>
      <button type="submit" name="reset">Start Over</button>
    </form>
  </body>
</html>
"""

app = Flask(__name__)
random.seed(42)
game = GameEngine(setup='auto', user_ai=n_step_ai)
game.setup_game()


@app.route("/", methods=["GET", "POST"])
def play_game():
    
    game_cookie = request.cookies.get("game_board")
    print(game_cookie)
    print(request.form)

    resp = make_response(render_template_string(TEXT, msg="HI"))
    # c = ",".join(map(str, ttt.board))
    # resp.set_cookie("game_board", c)
    return resp


if __name__ == "__main__":
    app.run(debug=True)