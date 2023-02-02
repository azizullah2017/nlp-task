from main import response_message
from flask import Flask, request, jsonify
app = Flask(__name__)



@app.route('/webhooks/rest/webhook/',methods = ['POST','GET'])
def hello_world():    
    content = request.json
    # print(content['message'])
    response = response_message(content['message'])
    return jsonify({"recipient_id":content['sender'],"text":response})

if __name__ == '__main__':
	app.run(host= '0.0.0.0',port='5005',debug=True)
