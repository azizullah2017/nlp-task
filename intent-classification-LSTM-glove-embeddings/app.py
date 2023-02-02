import pickle
from tensorflow.keras.models import load_model
from testmodel import IntentClassifier
from flask import Flask, request, jsonify
app = Flask(__name__)


model = load_model('./models/intents.h5')
with open('./utils/classes.pkl','rb') as file:
  classes = pickle.load(file)

with open('./utils/tokenizer.pkl','rb') as file:
  tokenizer = pickle.load(file)

with open('./utils/label_encoder.pkl','rb') as file:
  label_encoder = pickle.load(file)
     
nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
     
# print("========================")
# print(nlu.get_intent("is it cold in India right now"))
# print("========================")
# print(nlu.get_intent("how much tax is on my salary"))



@app.route('/webhooks/rest/webhook/',methods = ['POST','GET'])
def hello_world():    
    content = request.json
    # print(content['message'])
    response = nlu.get_intent(content['message'])
    return jsonify({"recipient_id":content['sender'],"text":response})

if __name__ == '__main__':
	app.run(host= '0.0.0.0',port='5005',debug=True)
