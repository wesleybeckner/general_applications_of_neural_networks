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
              {{"disabled" if board[j*3+i+1]!=" "}}>
              {{board[j*3+i+1]}}
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
game = GameEngine(setup='auto')
game.setup_game()

@app.route("/", methods=["GET", "POST"])
def play_game():
    
    game_cookie = request.cookies.get("game_board")
    print(game_cookie)
    # print(request.form['choice'])
    if game_cookie:
      game.board = {i: x for i, x in zip(range(1,10),
                    game_cookie.split(","))}
    if "choice" in request.form:
      move = int(request.form['choice'])
      winner, board = game.make_move(move)
      print(board)
    if "reset" in request.form:
      game.setup_game()
      game.winner = ""
      game.board = {i: " " for i in range(1,10)}
    if game.winner != "":
      if game.winner == "Stalemate":
        msg = game.winner + "!"
      else:
        msg = game.winner + " Won!"
    else:
      msg = "play move"
    resp = make_response(render_template_string(TEXT, 
                                                msg=msg, 
                                                board=game.board))
    c = ",".join(map(str, game.board.values()))
    resp.set_cookie("game_board", c)
    first_move = False
    return resp


if __name__ == "__main__":
    port = 5000 + random.randint(0, 999)
    print(port)
    url = "http://127.0.0.1:{0}".format(port)
    print(url)
    app.run(use_reloader=True, debug=True, port=port)