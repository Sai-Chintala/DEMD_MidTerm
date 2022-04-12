from flask import Flask

app = Flask(__name__)

@app.route('/')
def base_route():
    return "Welcome to Praxis"

@app.route('/<name>')
def print_name(name):
    return f"welcome {name}"

if __name__ == "__main__":
    app.run()